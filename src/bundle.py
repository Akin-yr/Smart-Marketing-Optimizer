import os, warnings, random
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sqlalchemy import create_engine
from xgboost import XGBRegressor
from prophet import Prophet
from scipy.optimize import minimize
from lunardate import LunarDate
import holidays

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

DB_URL  = os.getenv("DATABASE_URL", "postgresql://postgres:123456@localhost:5432/smart_marketing_db")
OUT_PKL = os.path.join(os.path.dirname(__file__), "model_bundle.pkl")

# 1.  LOAD DATA
def load_data(db_url: str):
    engine = create_engine(db_url)
    sales       = pd.read_sql("sales",             engine, parse_dates=["date"])
    submission  = pd.read_sql("sample_submission",  engine, parse_dates=["date"])
    orders      = pd.read_sql("orders",             engine, parse_dates=["order_date"])
    order_items = pd.read_sql("order_items",        engine)
    promotions  = pd.read_sql("promotions",         engine, parse_dates=["start_date","end_date"])
    web_traffic = pd.read_sql("web_traffic",        engine, parse_dates=["date"])
    inventory   = pd.read_sql("inventory",          engine, parse_dates=["snapshot_date"])
    return sales, submission, orders, order_items, promotions, web_traffic, inventory

# 2.  FEATURE ENGINEERING 
def build_tet_lookup():
    lookup = {}
    for y in range(2010, 2028):
        try:
            lookup[y] = pd.Timestamp(LunarDate(y, 1, 1).toSolarDate())
        except Exception:
            pass
    return lookup

TET_LOOKUP = build_tet_lookup()
VN_HOLIDAYS = holidays.country_holidays("VN", years=range(2012, 2026))


def tet_distance(date):
    candidates = [TET_LOOKUP[y] for y in [date.year-1, date.year, date.year+1] if y in TET_LOOKUP]
    diffs = [(date - tc).days for tc in candidates]
    return min(diffs, key=abs)


def add_calendar(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["year"]          = d["date"].dt.year
    d["month"]         = d["date"].dt.month
    d["day"]           = d["date"].dt.day
    d["dayofweek"]     = d["date"].dt.dayofweek
    d["dayofyear"]     = d["date"].dt.dayofyear
    d["weekofyear"]    = d["date"].dt.isocalendar().week.astype(int)
    d["quarter"]       = d["date"].dt.quarter
    d["week_in_month"] = ((d["day"] - 1) // 7) + 1
    d["is_weekend"]    = (d["dayofweek"] >= 5).astype(int)
    d["is_month_start"]= d["date"].dt.is_month_start.astype(int)
    d["is_month_end"]  = d["date"].dt.is_month_end.astype(int)
    d["is_holiday"]    = d["date"].apply(lambda x: int(x in VN_HOLIDAYS))

    d["is_1111"]        = ((d["month"]==11) & (d["day"]==11)).astype(int)
    d["is_1212"]        = ((d["month"]==12) & (d["day"]==12)).astype(int)
    d["is_99"]          = ((d["month"]==9)  & (d["day"]==9)).astype(int)
    d["is_1010"]        = ((d["month"]==10) & (d["day"]==10)).astype(int)
    d["is_blackfriday"] = ((d["month"]==11) & (d["dayofweek"]==4) & d["day"].between(23,29)).astype(int)
    d["is_event_window"]= (
        ((d["month"]==11) & d["day"].between(9,13)) |
        ((d["month"]==12) & d["day"].between(10,14)) |
        ((d["month"]==9)  & d["day"].between(7,11))
    ).astype(int)

    d["tet_distance"]       = d["date"].apply(tet_distance)
    d["tet_distance_abs"]   = d["tet_distance"].abs()
    d["tet_proximity"]      = np.exp(-d["tet_distance_abs"] / 10)
    d["tet_phase_pre21"]    = d["tet_distance"].between(-21, -8).astype(int)
    d["tet_phase_pre7"]     = d["tet_distance"].between(-7,  -1).astype(int)
    d["tet_phase_week"]     = d["tet_distance"].between( 0,   6).astype(int)
    d["tet_phase_post7"]    = d["tet_distance"].between( 7,  14).astype(int)
    d["tet_phase_far_after"]= d["tet_distance"].between(15,  35).astype(int)

    d["near_1111"]   = ((d["month"]==11) & d["day"].between(8,14)).astype(int)
    d["near_1212"]   = ((d["month"]==12) & d["day"].between(9,15)).astype(int)
    d["trend_lin"]   = (d["date"] - pd.Timestamp("2012-07-04")).dt.days / 365.0
    d["trend_sqrt"]  = np.sqrt(d["trend_lin"].clip(lower=0))
    d["trend_log"]   = np.log1p(d["trend_lin"].clip(lower=0))
    d["time_idx"]    = (d["date"] - d["date"].min()).dt.days
    return d


def fill_seasonal(df_in: pd.DataFrame, cols: list, growth_dampening=0.5) -> pd.DataFrame:
    df_out   = df_in.copy()
    train_part = df_out[df_out["is_test"] == 0]
    for c in cols:
        if c not in df_out.columns or df_out[c].isna().sum() == 0:
            continue
        season_profile = (
            train_part.groupby(["month","dayofweek"])[c].mean().reset_index()
            .rename(columns={c: c+"_seasonal"})
        )
        mean_overall = train_part[c].mean()
        yearly = train_part.groupby("year")[c].mean().dropna()
        if len(yearly) >= 3:
            growth_raw = (yearly.iloc[-1] / (yearly.iloc[0] + 1e-8)) ** (1.0 / (len(yearly)-1))
            growth = 1.0 + (growth_raw - 1.0) * growth_dampening
        else:
            growth = 1.0
        last_train_year = int(train_part["year"].max())
        df_out = df_out.merge(season_profile, on=["month","dayofweek"], how="left")
        years_ahead = (df_out["year"] - last_train_year).clip(lower=0)
        fill_values = df_out[c+"_seasonal"] * (growth ** years_ahead)
        df_out[c] = df_out[c].fillna(fill_values).fillna(mean_overall).fillna(0)
        df_out = df_out.drop(columns=[c+"_seasonal"])
    return df_out


def build_features(sales, submission, orders, order_items, promotions, web_traffic, inventory):
    """Toàn bộ feature engineering — trả về df, FEATURES list."""
    test_dates = submission[["date"]].copy().sort_values("date").reset_index(drop=True)

    df = pd.concat([
        sales[["date","revenue","cogs"]].assign(is_test=0),
        test_dates.assign(is_test=1, revenue=np.nan, cogs=np.nan)
    ], ignore_index=True).sort_values("date").drop_duplicates("date").reset_index(drop=True)

    df = add_calendar(df)

    # Promotions
    all_dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    promo_daily = pd.DataFrame({"date": all_dates})
    promo_daily["n_active_promos"]    = promo_daily["date"].apply(
        lambda d: int(((promotions["start_date"]<=d)&(promotions["end_date"]>=d)).sum()))
    promo_daily["avg_discount_value"] = promo_daily["date"].apply(
        lambda d: promotions.loc[(promotions["start_date"]<=d)&(promotions["end_date"]>=d),"discount_value"].mean())
    promo_daily["max_discount_value"] = promo_daily["date"].apply(
        lambda d: promotions.loc[(promotions["start_date"]<=d)&(promotions["end_date"]>=d),"discount_value"].max())
    promo_daily = promo_daily.fillna(0)
    df = df.merge(promo_daily, on="date", how="left")

    # Web traffic
    wt_daily = web_traffic.groupby("date").agg(
        sessions=("sessions","sum"), unique_visitors=("unique_visitors","sum"),
        page_views=("page_views","sum"), bounce_rate=("bounce_rate","mean"),
        avg_session_duration_sec=("avg_session_duration_sec","mean")
    ).reset_index()
    df = df.merge(wt_daily, on="date", how="left")
    EXO_WT = ["sessions","unique_visitors","page_views","bounce_rate","avg_session_duration_sec"]
    df = fill_seasonal(df, EXO_WT)

    # Orders (safe lags ≥549)
    orders_daily = orders.groupby("order_date").size().reset_index(name="_raw_n_orders").rename(columns={"order_date":"date"})
    oi_m = order_items.merge(orders[["order_id","order_date"]], on="order_id", how="left")
    items_daily = oi_m.groupby("order_date").agg(
        _raw_n_items=("quantity","sum"), _raw_n_unique_products=("product_id","nunique")
    ).reset_index().rename(columns={"order_date":"date"})
    df = df.merge(orders_daily, on="date", how="left").merge(items_daily, on="date", how="left")
    for c in ["_raw_n_orders","_raw_n_items","_raw_n_unique_products"]:
        df[c] = df[c].fillna(0)
    for col, short in [("_raw_n_orders","ord"),("_raw_n_items","itm"),("_raw_n_unique_products","upd")]:
        for lag in [549,730,1095]:
            df[f"{short}_lag_{lag}"] = df[col].shift(lag)
        shifted = df[col].shift(549)
        df[f"{short}_rollmean_30_l549"] = shifted.rolling(30, min_periods=1).mean()
        df[f"{short}_rollmean_90_l549"] = shifted.rolling(90, min_periods=1).mean()
    df = df.drop(columns=["_raw_n_orders","_raw_n_items","_raw_n_unique_products"])

    # Inventory
    inv_agg = inventory.groupby("snapshot_date").agg(
        inv_stockout_flag=("stockout_flag","max"), inv_fill_rate=("fill_rate","mean"),
        inv_sell_through=("sell_through_rate","mean"), inv_days_supply=("days_of_supply","mean"),
        inv_stockout_days=("stockout_days","mean"), inv_overstock_flag=("overstock_flag","max"),
    ).reset_index().rename(columns={"snapshot_date":"date"})
    inv_daily = pd.DataFrame({"date":all_dates}).merge(inv_agg, on="date", how="left").sort_values("date").reset_index(drop=True)
    INV_COLS = [c for c in inv_agg.columns if c != "date"]
    for c in INV_COLS: inv_daily[c] = inv_daily[c].ffill().bfill()
    df = df.merge(inv_daily, on="date", how="left")
    df = fill_seasonal(df, INV_COLS)

    # Target lags & rolling
    df = df.sort_values("date").reset_index(drop=True)
    SAFE_TARGET_LAGS = [549,600,700,730,1095,1460]
    BASE_LAG = 549
    for target in ["revenue","cogs"]:
        for lag in SAFE_TARGET_LAGS:
            df[f"{target}_lag_{lag}"] = df[target].shift(lag)
        shifted = df[target].shift(BASE_LAG)
        for w in [7,14,30,60,90,180]:
            df[f"{target}_rollmean_{w}_l{BASE_LAG}"] = shifted.rolling(w, min_periods=1).mean()
            df[f"{target}_rollstd_{w}_l{BASE_LAG}"]  = shifted.rolling(w, min_periods=1).std()
            df[f"{target}_rollmax_{w}_l{BASE_LAG}"]  = shifted.rolling(w, min_periods=1).max()

    EXO_SHORT = ["sessions","unique_visitors","n_active_promos","avg_discount_value"]
    for col in EXO_SHORT:
        for lag in [1,7,14,28]:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        df[f"{col}_rollmean_7"]  = df[col].shift(1).rolling(7,  min_periods=1).mean()
        df[f"{col}_rollmean_28"] = df[col].shift(1).rolling(28, min_periods=1).mean()

    for target in ["revenue","cogs"]:
        for lag in [729,731,728,732,735,721]:
            df[f"{target}_lag_{lag}"] = df[target].shift(lag)
        cluster_cols = [f"{target}_lag_{l}" for l in [728,729,730,731,732]]
        df[f"{target}_lag_730_med"]  = df[cluster_cols].median(axis=1)
        df[f"{target}_lag_730_mean"] = df[cluster_cols].mean(axis=1)
        df[f"{target}_lag_730_std"]  = df[cluster_cols].std(axis=1)
        multi_yr = [f"{target}_lag_{l}" for l in [730,1095,1460]]
        df[f"{target}_multi_yr_mean"]   = df[multi_yr].mean(axis=1)
        df[f"{target}_multi_yr_median"] = df[multi_yr].median(axis=1)
        df[f"{target}_trend_recent"] = df[f"{target}_lag_730"] / (df[f"{target}_lag_1095"] + 1e-8)
        df[f"{target}_trend_old"]    = df[f"{target}_lag_1095"] / (df[f"{target}_lag_1460"] + 1e-8)

    def compute_holiday_adj_factor(target):
        train_only = df[df["is_test"]==0].copy()
        train_only["_baseline"] = train_only[target].rolling(7, min_periods=1).mean()
        train_only["_ratio"]    = train_only[target] / (train_only["_baseline"] + 1e-8)
        factors = {}
        for flag_col in ["is_1111","is_1212","is_99","is_1010","is_blackfriday",
                         "tet_phase_pre7","tet_phase_week","tet_phase_far_after"]:
            if flag_col in train_only.columns:
                factors[flag_col] = train_only.loc[train_only[flag_col]==1,"_ratio"].median()
        return factors

    for target in ["revenue","cogs"]:
        factors = compute_holiday_adj_factor(target)
        df[f"{target}_holiday_adj"] = 1.0
        for flag_col, fac in factors.items():
            if not np.isnan(fac):
                df.loc[df[flag_col]==1, f"{target}_holiday_adj"] = fac

    df["is_peak_season"] = df["month"].isin([3,4,5,6]).astype(int)
    df["margin_ratio_l549"] = df["revenue"].shift(549) / (df["cogs"].shift(549) + 1e-8)
    df["margin_ratio_l730"] = df["revenue"].shift(730) / (df["cogs"].shift(730) + 1e-8)

    # Seasonal profiles (from train only)
    train_part = df[df["is_test"]==0].copy()
    for target in ["revenue","cogs"]:
        for grp, name, new_col in [
            (["month","dayofweek"],     None,               f"{target}_profile_mdow"),
            (["dayofyear"],             None,               f"{target}_profile_doy"),
            (["month","week_in_month","dayofweek"], None,   f"{target}_profile_mwdow"),
        ]:
            prof = train_part.groupby(grp)[target].mean().reset_index().rename(columns={target: new_col})
            df = df.merge(prof, on=grp, how="left")
        df[f"{target}_yoy_delta_730_1095"] = df[f"{target}_lag_730"] - df[f"{target}_lag_1095"]
        df[f"{target}_yoy_ratio_730_1095"] = df[f"{target}_lag_730"] / (df[f"{target}_lag_1095"] + 1e-8)

    df["promo_x_sessions"] = df["n_active_promos_lag_1"] * df["sessions_lag_1"]
    df["promo_x_tet"]      = df["n_active_promos"]       * df["tet_proximity"]
    df["discount_x_tet"]   = df["avg_discount_value"]    * df["tet_proximity"]
    df["event_x_sessions"] = df["is_event_window"]       * df["sessions_lag_1"]

    # Trend polynomial
    tr_mask = df["is_test"] == 0
    valid_idx = df.dropna(subset=["revenue_lag_1095"]).index
    full_train_mask = tr_mask & df.index.isin(valid_idx)

    for target, attr in [("revenue","trend_coefs_rev"), ("cogs","trend_coefs_cog")]:
        x = df.loc[full_train_mask, "time_idx"].values
        y = np.log1p(df.loc[full_train_mask, target].values)
        coefs = np.polyfit(x, y, 2)
        df[f"trend_pred_{target[:3]}_log"] = np.polyval(coefs, df["time_idx"].values)

    # Feature list
    toxic_lags   = [c for c in df.columns if ("revenue_lag_" in c or "cogs_lag_" in c)]
    safe_m5      = [c for c in toxic_lags if any(k in c for k in ["mean","med","std","trend"])]
    lags_to_drop = [c for c in toxic_lags if c not in safe_m5]
    sin_cos_drop = [c for c in df.columns if any(c.endswith(s) for s in ["_sin","_cos"])]
    NON_FEATURES = ["date","revenue","cogs","is_test","tet_distance","year"] + lags_to_drop + sin_cos_drop
    FEATURES = [c for c in df.columns if c not in NON_FEATURES]

    return df, FEATURES, full_train_mask

# 3.  TRAINING HELPERS
SEEDS = [42, 7, 123, 2024, 777]
LOG_TARGETS = True

def fwd(y): return np.log1p(y) if LOG_TARGETS else y
def back(y): return np.clip(np.expm1(y) if LOG_TARGETS else y, 0, None)

LGB_MAE = dict(objective="regression_l1", metric="mae", learning_rate=0.025,
               num_leaves=63, min_data_in_leaf=30, feature_fraction=0.7,
               bagging_fraction=0.8, bagging_freq=5, lambda_l1=0.3, lambda_l2=0.3, verbose=-1)

LGB_Q90 = dict(objective="quantile", metric="quantile", alpha=0.90, learning_rate=0.025,
               num_leaves=63, min_data_in_leaf=30, feature_fraction=0.7,
               bagging_fraction=0.8, bagging_freq=5, lambda_l1=0.3, lambda_l2=0.3, verbose=-1)

LGB_TWEEDIE = dict(objective="tweedie", tweedie_variance_power=1.5, metric="mae",
                   learning_rate=0.025, num_leaves=63, min_data_in_leaf=30,
                   feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
                   lambda_l1=0.3, lambda_l2=0.3, verbose=-1)

CATEGORICAL = ["month","dayofweek","quarter","week_in_month","dayofyear","weekofyear"]


def train_lgb_full(df, mask, features, target, params, n_iters, use_log=True):
    X = df.loc[mask, features]
    y = fwd(df.loc[mask, target].values) if use_log else df.loc[mask, target].values
    w = df.loc[mask, "recency_weight"].values
    cats = [c for c in CATEGORICAL if c in features]
    models = []
    for seed in SEEDS:
        p = {**params, "seed": seed, "feature_fraction_seed": seed, "bagging_seed": seed}
        m = lgb.train(p, lgb.Dataset(X, y, weight=w, categorical_feature=cats),
                      num_boost_round=int(n_iters * 1.1))
        models.append(m)
    return models


def train_xgb_full(df, mask, features, target, n_iters):
    X = df.loc[mask, features].values
    y = fwd(df.loc[mask, target].values)
    w = df.loc[mask, "recency_weight"].values
    models = []
    for seed in SEEDS:
        m = XGBRegressor(n_estimators=int(n_iters*1.1), learning_rate=0.025, max_depth=7,
                         min_child_weight=8, subsample=0.8, colsample_bytree=0.7,
                         reg_alpha=0.3, reg_lambda=0.3, random_state=seed,
                         tree_method="hist", objective="reg:absoluteerror", verbosity=0)
        m.fit(X, y, sample_weight=w, verbose=False)
        models.append(m)
    return models


def build_tet_holidays_df():
    return pd.DataFrame({
        "holiday": "tet",
        "ds": list(TET_LOOKUP.values()),
        "lower_window": -7,
        "upper_window": 7,
    })


PROPHET_REGRESSORS = ["sessions","n_active_promos","tet_proximity"]


def train_prophet_model(df, mask, target):
    fit_df = df.loc[mask, ["date",target]+PROPHET_REGRESSORS].rename(columns={"date":"ds",target:"y"}).copy()
    fit_df["y"] = np.log1p(fit_df["y"])
    m = Prophet(yearly_seasonality=20, weekly_seasonality=True, daily_seasonality=False,
                changepoint_prior_scale=0.05, seasonality_prior_scale=20.0,
                holidays=build_tet_holidays_df(), seasonality_mode="additive")
    m.add_country_holidays(country_name="VN")
    for c in PROPHET_REGRESSORS: m.add_regressor(c)
    m.fit(fit_df)
    return m


def train_hybrid_resid(df, prophet_model, features, fit_mask, val_mask, target):
    future_fit = df.loc[fit_mask, ["date"]+PROPHET_REGRESSORS].rename(columns={"date":"ds"})
    prop_fit   = back(prophet_model.predict(future_fit)["yhat"].values)
    resid_fit  = df.loc[fit_mask, target].values - prop_fit

    future_val = df.loc[val_mask, ["date"]+PROPHET_REGRESSORS].rename(columns={"date":"ds"})
    prop_val   = back(prophet_model.predict(future_val)["yhat"].values)
    resid_val  = df.loc[val_mask, target].values - prop_val

    X_fit = df.loc[fit_mask, features]; w_fit = df.loc[fit_mask, "recency_weight"].values
    X_val = df.loc[val_mask, features]
    cats  = [c for c in CATEGORICAL if c in features]

    params = dict(objective="regression", metric="mae", learning_rate=0.02, num_leaves=63,
                  min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                  lambda_l1=0.1, lambda_l2=0.1, verbose=-1, seed=SEED)
    resid_model = lgb.train(params, lgb.Dataset(X_fit, resid_fit, weight=w_fit, categorical_feature=cats),
                            num_boost_round=4000,
                            valid_sets=[lgb.Dataset(X_val, resid_val, reference=lgb.Dataset(X_fit, resid_fit))],
                            valid_names=["val"],
                            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    return resid_model


def optimise_weights(y, preds_list):
    P = np.column_stack(preds_list); n = P.shape[1]
    def obj(w):
        w = np.clip(w, 0, None); s = w.sum()
        if s < 1e-9: return 1e18
        return np.mean(np.abs(P @ (w/s) - y))
    res = minimize(obj, np.ones(n)/n, method="Nelder-Mead",
                   options={"xatol":1e-5,"fatol":1e-5,"maxiter":10000})
    w = np.clip(res.x, 0, None); s = w.sum()
    return w/s if s > 0 else np.ones(n)/n


# 4.  MAIN — TRAIN & SAVE
def main():
    print("Loading data")
    sales, submission, orders, order_items, promotions, web_traffic, inventory = load_data(DB_URL)

    print("Building features")
    df, FEATURES, full_train_mask = build_features(sales, submission, orders, order_items, promotions, web_traffic, inventory)

    # Recency weights
    HALFLIFE_DAYS = 20 * 365
    max_train_date = df.loc[full_train_mask, "date"].max()
    days_from_end  = (max_train_date - df["date"]).dt.days.clip(lower=0)
    df["recency_weight"] = np.exp(-np.log(2) * days_from_end / HALFLIFE_DAYS)

    # Val split để tìm optimal weights
    VAL_START  = pd.Timestamp("2022-01-01")
    valid_idx  = df.dropna(subset=["revenue_lag_1095"]).index
    tr_mask    = (df["is_test"]==0) & (df["date"] <  VAL_START) & df.index.isin(valid_idx)
    val_mask   = (df["is_test"]==0) & (df["date"] >= VAL_START) & df.index.isin(valid_idx)

    print("Training base models (val split) for weight search")
    # LGB-MAE
    def train_lgb_val(target, params, use_log=True):
        X_tr=df.loc[tr_mask,FEATURES]; y_tr=fwd(df.loc[tr_mask,target].values) if use_log else df.loc[tr_mask,target].values
        X_va=df.loc[val_mask,FEATURES]; y_va=fwd(df.loc[val_mask,target].values) if use_log else df.loc[val_mask,target].values
        w_tr=df.loc[tr_mask,"recency_weight"].values; w_va=df.loc[val_mask,"recency_weight"].values
        cats=[c for c in CATEGORICAL if c in FEATURES]
        val_preds, best_iters = [], []
        for seed in SEEDS:
            p={**params,"seed":seed,"feature_fraction_seed":seed,"bagging_seed":seed}
            dtr=lgb.Dataset(X_tr,y_tr,weight=w_tr,categorical_feature=cats)
            dva=lgb.Dataset(X_va,y_va,weight=w_va,reference=dtr,categorical_feature=cats)
            m=lgb.train(p,dtr,num_boost_round=8000,valid_sets=[dva],valid_names=["val"],
                        callbacks=[lgb.early_stopping(300,verbose=False),lgb.log_evaluation(0)])
            pred = np.clip(m.predict(X_va,num_iteration=m.best_iteration),0,None)
            val_preds.append(back(pred) if use_log else pred)
            best_iters.append(m.best_iteration)
        return np.mean(val_preds,axis=0), int(np.mean(best_iters))

    pred_rev_lgb_val, bi_rev_lgb = train_lgb_val("revenue", LGB_MAE)
    pred_cog_lgb_val, bi_cog_lgb = train_lgb_val("cogs",    LGB_MAE)
    pred_rev_q90_val, bi_rev_q90 = train_lgb_val("revenue", LGB_Q90)
    pred_cog_q90_val, bi_cog_q90 = train_lgb_val("cogs",    LGB_Q90)
    pred_rev_tw_val,  bi_rev_tw  = train_lgb_val("revenue", LGB_TWEEDIE, use_log=False)
    pred_cog_tw_val,  bi_cog_tw  = train_lgb_val("cogs",    LGB_TWEEDIE, use_log=False)

    def train_xgb_val(target):
        X_tr=df.loc[tr_mask,FEATURES].values; y_tr=fwd(df.loc[tr_mask,target].values)
        X_va=df.loc[val_mask,FEATURES].values; y_va=fwd(df.loc[val_mask,target].values)
        w_tr=df.loc[tr_mask,"recency_weight"].values; w_va=df.loc[val_mask,"recency_weight"].values
        val_preds, best_iters = [], []
        for seed in SEEDS:
            m=XGBRegressor(n_estimators=8000,learning_rate=0.025,max_depth=7,min_child_weight=8,
                           subsample=0.8,colsample_bytree=0.7,reg_alpha=0.3,reg_lambda=0.3,
                           early_stopping_rounds=300,random_state=seed,tree_method="hist",
                           objective="reg:absoluteerror",eval_metric="mae",verbosity=0)
            m.fit(X_tr,y_tr,sample_weight=w_tr,eval_set=[(X_va,y_va)],
                  sample_weight_eval_set=[w_va],verbose=False)
            val_preds.append(back(m.predict(X_va))); best_iters.append(m.best_iteration)
        return np.mean(val_preds,axis=0), int(np.mean(best_iters))

    pred_rev_xgb_val, bi_rev_xgb = train_xgb_val("revenue")
    pred_cog_xgb_val, bi_cog_xgb = train_xgb_val("cogs")

    print("Training Prophet + Hybrid (val)")
    prophet_rev_val = train_prophet_model(df, tr_mask, "revenue")
    prophet_cog_val = train_prophet_model(df, tr_mask, "cogs")

    pseudo_val_start = df.loc[tr_mask,"date"].quantile(0.9)
    ps_tr = tr_mask & (df["date"] < pseudo_val_start)
    ps_va = tr_mask & (df["date"] >= pseudo_val_start)
    resid_rev_val = train_hybrid_resid(df, prophet_rev_val, FEATURES, ps_tr, ps_va, "revenue")
    resid_cog_val = train_hybrid_resid(df, prophet_cog_val, FEATURES, ps_tr, ps_va, "cogs")

    future_val = df.loc[val_mask,["date"]+PROPHET_REGRESSORS].rename(columns={"date":"ds"})
    pred_rev_hyb_val = np.clip(back(prophet_rev_val.predict(future_val)["yhat"].values)
                               + resid_rev_val.predict(df.loc[val_mask,FEATURES], num_iteration=resid_rev_val.best_iteration), 0, None)
    pred_cog_hyb_val = np.clip(back(prophet_cog_val.predict(future_val)["yhat"].values)
                               + resid_cog_val.predict(df.loc[val_mask,FEATURES], num_iteration=resid_cog_val.best_iteration), 0, None)

    # Optimal weights
    y_rev_val = df.loc[val_mask,"revenue"].values
    y_cog_val = df.loc[val_mask,"cogs"].values
    w_rev = optimise_weights(y_rev_val, [pred_rev_lgb_val, pred_rev_xgb_val, pred_rev_hyb_val, pred_rev_tw_val])
    w_cog = optimise_weights(y_cog_val, [pred_cog_lgb_val, pred_cog_xgb_val, pred_cog_hyb_val, pred_cog_tw_val])
    print(f"Optimal weights REV: {w_rev}")
    print(f"Optimal weights COG: {w_cog}")

    print("Retraining on full train data")
    models_rev_lgb = train_lgb_full(df, full_train_mask, FEATURES, "revenue", LGB_MAE, bi_rev_lgb)
    models_cog_lgb = train_lgb_full(df, full_train_mask, FEATURES, "cogs",    LGB_MAE, bi_cog_lgb)
    models_rev_q90 = train_lgb_full(df, full_train_mask, FEATURES, "revenue", LGB_Q90, bi_rev_q90)
    models_cog_q90 = train_lgb_full(df, full_train_mask, FEATURES, "cogs",    LGB_Q90, bi_cog_q90)
    models_rev_tw  = train_lgb_full(df, full_train_mask, FEATURES, "revenue", LGB_TWEEDIE, bi_rev_tw, use_log=False)
    models_cog_tw  = train_lgb_full(df, full_train_mask, FEATURES, "cogs",    LGB_TWEEDIE, bi_cog_tw, use_log=False)
    models_rev_xgb = train_xgb_full(df, full_train_mask, FEATURES, "revenue", bi_rev_xgb)
    models_cog_xgb = train_xgb_full(df, full_train_mask, FEATURES, "cogs",    bi_cog_xgb)

    prophet_rev_full = train_prophet_model(df, full_train_mask, "revenue")
    prophet_cog_full = train_prophet_model(df, full_train_mask, "cogs")
    pv_start = df.loc[full_train_mask,"date"].quantile(0.9)
    ftr_mask = full_train_mask & (df["date"] <  pv_start)
    fva_mask = full_train_mask & (df["date"] >= pv_start)
    resid_rev_full = train_hybrid_resid(df, prophet_rev_full, FEATURES, ftr_mask, fva_mask, "revenue")
    resid_cog_full = train_hybrid_resid(df, prophet_cog_full, FEATURES, ftr_mask, fva_mask, "cogs")

    # Trend coefs (baked into df already, just need values for test-time reconstruction)
    trend_coefs_rev = np.polyfit(df.loc[full_train_mask,"time_idx"].values,
                                 np.log1p(df.loc[full_train_mask,"revenue"].values), 2)
    trend_coefs_cog = np.polyfit(df.loc[full_train_mask,"time_idx"].values,
                                 np.log1p(df.loc[full_train_mask,"cogs"].values), 2)

    # Save bundle
    bundle = {
        # Models
        "models_rev_lgb": models_rev_lgb,
        "models_cog_lgb": models_cog_lgb,
        "models_rev_q90": models_rev_q90,
        "models_cog_q90": models_cog_q90,
        "models_rev_tw":  models_rev_tw,
        "models_cog_tw":  models_cog_tw,
        "models_rev_xgb": models_rev_xgb,
        "models_cog_xgb": models_cog_xgb,
        "prophet_rev":    prophet_rev_full,
        "prophet_cog":    prophet_cog_full,
        "resid_rev":      resid_rev_full,
        "resid_cog":      resid_cog_full,
        # Meta
        "FEATURES":       FEATURES,
        "w_rev":          w_rev,
        "w_cog":          w_cog,
        "trend_coefs_rev": trend_coefs_rev,
        "trend_coefs_cog": trend_coefs_cog,
        "time_idx_origin": df["date"].min(),   # cần để tính time_idx cho ngày mới
        "train_df_tail":  df[df["is_test"]==0][["date","revenue","cogs"]].tail(1500),  # cho lag lookback
    }

    joblib.dump(bundle, OUT_PKL, compress=3)
    print(f"\nBundle saved → {OUT_PKL}")
    print(f"ize: {os.path.getsize(OUT_PKL)/1e6:.1f} MB")


if __name__ == "__main__":
    main()