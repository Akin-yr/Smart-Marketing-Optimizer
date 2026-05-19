import warnings, os
import numpy as np
import pandas as pd
import holidays
from lunardate import LunarDate
from sqlalchemy import create_engine

warnings.filterwarnings("ignore")

LOG_TARGETS = True
Q_BLEND     = 0.30
TREND_BLEND = 0.10
SHIFT_CONST = 1.14

TET_LOOKUP = {}
for _y in range(2010, 2030):
    try:
        TET_LOOKUP[_y] = pd.Timestamp(LunarDate(_y, 1, 1).toSolarDate())
    except Exception:
        pass

VN_HOLIDAYS = holidays.country_holidays("VN", years=range(2012, 2030))
PROPHET_REGRESSORS = ["sessions", "n_active_promos", "tet_proximity"]
CATEGORICAL = ["month","dayofweek","quarter","week_in_month","dayofyear","weekofyear"]


def back(y): return np.clip(np.expm1(y) if LOG_TARGETS else y, 0, None)


def tet_distance(date):
    candidates = [TET_LOOKUP[y] for y in [date.year-1, date.year, date.year+1] if y in TET_LOOKUP]
    return min(((date - tc).days for tc in candidates), key=abs)


def add_calendar(d: pd.DataFrame, time_idx_origin: pd.Timestamp) -> pd.DataFrame:
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
    d["tet_phase_pre21"]    = d["tet_distance"].between(-21,-8).astype(int)
    d["tet_phase_pre7"]     = d["tet_distance"].between(-7, -1).astype(int)
    d["tet_phase_week"]     = d["tet_distance"].between( 0,  6).astype(int)
    d["tet_phase_post7"]    = d["tet_distance"].between( 7, 14).astype(int)
    d["tet_phase_far_after"]= d["tet_distance"].between(15, 35).astype(int)

    d["near_1111"]   = ((d["month"]==11) & d["day"].between(8,14)).astype(int)
    d["near_1212"]   = ((d["month"]==12) & d["day"].between(9,15)).astype(int)
    d["trend_lin"]   = (d["date"] - pd.Timestamp("2012-07-04")).dt.days / 365.0
    d["trend_sqrt"]  = np.sqrt(d["trend_lin"].clip(lower=0))
    d["trend_log"]   = np.log1p(d["trend_lin"].clip(lower=0))
    d["time_idx"]    = (d["date"] - time_idx_origin).dt.days
    return d


def fill_seasonal_from_profile(df_in, profile_map):
    df_out = df_in.copy()
    for c, prof in profile_map.items():
        if c not in df_out.columns:
            continue
        df_out = df_out.merge(prof, on=["month","dayofweek"], how="left")
        df_out[c] = df_out[c].fillna(df_out[c+"_seasonal"]).fillna(0)
        df_out = df_out.drop(columns=[c+"_seasonal"], errors="ignore")
    return df_out

def predict_model(
    start_date: pd.Timestamp,
    end_date:   pd.Timestamp,
    db_url:     str,
    bundle:     dict,
) -> pd.DataFrame:
    """
    Trả về DataFrame với cột: date, revenue, cogs

    Parameters
    ----------
    start_date : ngày bắt đầu dự báo
    end_date   : ngày kết thúc dự báo
    db_url     : PostgreSQL connection string
    bundle     : dict được load từ model_bundle.pkl
    """
    # ── 1. Unpack bundle ──────────────────────────────────────
    FEATURES         = bundle["FEATURES"]
    w_rev            = bundle["w_rev"]
    w_cog            = bundle["w_cog"]
    trend_coefs_rev  = bundle["trend_coefs_rev"]
    trend_coefs_cog  = bundle["trend_coefs_cog"]
    time_idx_origin  = bundle["time_idx_origin"]
    train_tail       = bundle["train_df_tail"].copy()   # DataFrame [date, revenue, cogs]

    models_rev_lgb = bundle["models_rev_lgb"]
    models_cog_lgb = bundle["models_cog_lgb"]
    models_rev_q90 = bundle["models_rev_q90"]
    models_cog_q90 = bundle["models_cog_q90"]
    models_rev_tw  = bundle["models_rev_tw"]
    models_cog_tw  = bundle["models_cog_tw"]
    models_rev_xgb = bundle["models_rev_xgb"]
    models_cog_xgb = bundle["models_cog_xgb"]
    prophet_rev    = bundle["prophet_rev"]
    prophet_cog    = bundle["prophet_cog"]
    resid_rev      = bundle["resid_rev"]
    resid_cog      = bundle["resid_cog"]

    # 2. Load live data from DB 
    engine = create_engine(db_url)
    promotions  = pd.read_sql("promotions",  engine, parse_dates=["start_date","end_date"])
    web_traffic = pd.read_sql("web_traffic", engine, parse_dates=["date"])
    inventory   = pd.read_sql("inventory",   engine, parse_dates=["snapshot_date"])

    # 3. Build prediction frame
    test_dates = pd.date_range(start_date, end_date, freq="D")
    # Pad lookback: cần tối thiểu 1460 ngày trước start_date để tính lags
    LOOKBACK_DAYS = 1500
    lookback_start = start_date - pd.Timedelta(days=LOOKBACK_DAYS)
    lookback_dates = pd.date_range(lookback_start, start_date - pd.Timedelta(days=1), freq="D")

    # Lấy revenue/cogs lịch sử từ train_tail (đã baked trong bundle)
    hist = (train_tail
            .set_index("date")
            .reindex(lookback_dates)
            .reset_index()
            .rename(columns={"index":"date"}))
    hist["is_test"] = 0

    pred_frame = pd.DataFrame({"date": test_dates, "revenue": np.nan, "cogs": np.nan, "is_test": 1})
    df = pd.concat([hist, pred_frame], ignore_index=True).sort_values("date").reset_index(drop=True)
    df = add_calendar(df, time_idx_origin)

    # 4. Promotions
    all_dates = df["date"]
    promo_daily = pd.DataFrame({"date": all_dates})
    promo_daily["n_active_promos"] = promo_daily["date"].apply(
        lambda d: int(((promotions["start_date"]<=d)&(promotions["end_date"]>=d)).sum()))
    promo_daily["avg_discount_value"] = promo_daily["date"].apply(
        lambda d: promotions.loc[(promotions["start_date"]<=d)&(promotions["end_date"]>=d),"discount_value"].mean())
    promo_daily["max_discount_value"] = promo_daily["date"].apply(
        lambda d: promotions.loc[(promotions["start_date"]<=d)&(promotions["end_date"]>=d),"discount_value"].max())
    promo_daily = promo_daily.fillna(0)
    df = df.merge(promo_daily, on="date", how="left")

    # 5. Web traffic (aggregate + seasonal fill for test rows)
    wt_daily = web_traffic.groupby("date").agg(
        sessions=("sessions","sum"), unique_visitors=("unique_visitors","sum"),
        page_views=("page_views","sum"), bounce_rate=("bounce_rate","mean"),
        avg_session_duration_sec=("avg_session_duration_sec","mean")
    ).reset_index()
    df = df.merge(wt_daily, on="date", how="left")

    EXO_WT = ["sessions","unique_visitors","page_views","bounce_rate","avg_session_duration_sec"]
    # Simple forward-fill + seasonal mean for NaN (test rows)
    for c in EXO_WT:
        df[c] = df[c].ffill()
        if df[c].isna().any():
            col_mean = df.loc[df["is_test"]==0, c].mean()
            df[c] = df[c].fillna(col_mean)

    # 6. Inventory
    inv_agg = inventory.groupby("snapshot_date").agg(
        inv_stockout_flag=("stockout_flag","max"), inv_fill_rate=("fill_rate","mean"),
        inv_sell_through=("sell_through_rate","mean"), inv_days_supply=("days_of_supply","mean"),
        inv_stockout_days=("stockout_days","mean"), inv_overstock_flag=("overstock_flag","max"),
    ).reset_index().rename(columns={"snapshot_date":"date"})
    inv_daily = pd.DataFrame({"date":all_dates}).merge(inv_agg, on="date", how="left").sort_values("date").reset_index(drop=True)
    INV_COLS = [c for c in inv_agg.columns if c != "date"]
    for c in INV_COLS: inv_daily[c] = inv_daily[c].ffill().bfill()
    df = df.merge(inv_daily, on="date", how="left")
    for c in INV_COLS: df[c] = df[c].ffill().fillna(0)

    # 7. Lag / rolling features 
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
        for lag in [1,7,14,28]: df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        df[f"{col}_rollmean_7"]  = df[col].shift(1).rolling(7,  min_periods=1).mean()
        df[f"{col}_rollmean_28"] = df[col].shift(1).rolling(28, min_periods=1).mean()

    for target in ["revenue","cogs"]:
        for lag in [729,731,728,732,735,721]: df[f"{target}_lag_{lag}"] = df[target].shift(lag)
        cluster = [f"{target}_lag_{l}" for l in [728,729,730,731,732]]
        df[f"{target}_lag_730_med"]  = df[cluster].median(axis=1)
        df[f"{target}_lag_730_mean"] = df[cluster].mean(axis=1)
        df[f"{target}_lag_730_std"]  = df[cluster].std(axis=1)
        multi_yr = [f"{target}_lag_{l}" for l in [730,1095,1460]]
        df[f"{target}_multi_yr_mean"]   = df[multi_yr].mean(axis=1)
        df[f"{target}_multi_yr_median"] = df[multi_yr].median(axis=1)
        df[f"{target}_trend_recent"] = df[f"{target}_lag_730"] / (df[f"{target}_lag_1095"] + 1e-8)
        df[f"{target}_trend_old"]    = df[f"{target}_lag_1095"] / (df[f"{target}_lag_1460"] + 1e-8)
        df[f"{target}_holiday_adj"] = 1.0  # no recalc needed at inference
        df[f"{target}_yoy_delta_730_1095"] = df[f"{target}_lag_730"] - df[f"{target}_lag_1095"]
        df[f"{target}_yoy_ratio_730_1095"] = df[f"{target}_lag_730"] / (df[f"{target}_lag_1095"] + 1e-8)

    df["margin_ratio_l549"] = df["revenue"].shift(549) / (df["cogs"].shift(549) + 1e-8)
    df["margin_ratio_l730"] = df["revenue"].shift(730) / (df["cogs"].shift(730) + 1e-8)
    df["is_peak_season"]    = df["month"].isin([3,4,5,6]).astype(int)
    df["promo_x_sessions"]  = df["n_active_promos_lag_1"] * df["sessions_lag_1"]
    df["promo_x_tet"]       = df["n_active_promos"]       * df["tet_proximity"]
    df["discount_x_tet"]    = df["avg_discount_value"]    * df["tet_proximity"]
    df["event_x_sessions"]  = df["is_event_window"]       * df["sessions_lag_1"]

    # Seasonal profiles — use bundle's train_tail for proxy (simplified)
    for target in ["revenue","cogs"]:
        for col in [f"{target}_profile_mdow", f"{target}_profile_doy", f"{target}_profile_mwdow"]:
            if col not in df.columns: df[col] = 0.0

    # Trend features
    df["trend_pred_rev_log"] = np.polyval(trend_coefs_rev, df["time_idx"].values)
    df["trend_pred_cog_log"] = np.polyval(trend_coefs_cog, df["time_idx"].values)

    # 8. Slice test rows only
    test_mask = df["is_test"] == 1

    # Ensure all FEATURES exist (add missing as 0)
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0.0

    X_test    = df.loc[test_mask, FEATURES]
    X_test_np = X_test.values
    future    = df.loc[test_mask, ["date"]+PROPHET_REGRESSORS].rename(columns={"date":"ds"})

    # 9. Predict
    def avg_lgb(models, X):
        return np.mean([back(m.predict(X)) for m in models], axis=0)

    def avg_lgb_raw(models, X):
        return np.mean([np.clip(m.predict(X), 0, None) for m in models], axis=0)

    def avg_xgb(models, X):
        return np.mean([back(m.predict(X)) for m in models], axis=0)

    pred_rev_lgb = avg_lgb(models_rev_lgb, X_test)
    pred_cog_lgb = avg_lgb(models_cog_lgb, X_test)
    pred_rev_xgb = avg_xgb(models_rev_xgb, X_test_np)
    pred_cog_xgb = avg_xgb(models_cog_xgb, X_test_np)
    pred_rev_tw  = avg_lgb_raw(models_rev_tw, X_test)
    pred_cog_tw  = avg_lgb_raw(models_cog_tw, X_test)
    pred_rev_q90 = avg_lgb(models_rev_q90, X_test)
    pred_cog_q90 = avg_lgb(models_cog_q90, X_test)

    prop_rev = back(prophet_rev.predict(future)["yhat"].values)
    prop_cog = back(prophet_cog.predict(future)["yhat"].values)
    pred_rev_hyb = np.clip(prop_rev + resid_rev.predict(X_test, num_iteration=resid_rev.best_iteration), 0, None)
    pred_cog_hyb = np.clip(prop_cog + resid_cog.predict(X_test, num_iteration=resid_cog.best_iteration), 0, None)

    # 10. Ensemble
    base_rev = sum(wi*pi for wi,pi in zip(w_rev, [pred_rev_lgb, pred_rev_xgb, pred_rev_hyb, pred_rev_tw]))
    base_cog = sum(wi*pi for wi,pi in zip(w_cog, [pred_cog_lgb, pred_cog_xgb, pred_cog_hyb, pred_cog_tw]))

    pred_rev = (1 - Q_BLEND) * base_rev + Q_BLEND * pred_rev_q90
    pred_cog = (1 - Q_BLEND) * base_cog + Q_BLEND * pred_cog_q90

    # 11. Post-processing
    pred_rev = np.clip(pred_rev, 0, None)
    pred_cog = np.clip(pred_cog, 0, None)

    trend_rev = np.expm1(df.loc[test_mask, "trend_pred_rev_log"].values)
    trend_cog = np.expm1(df.loc[test_mask, "trend_pred_cog_log"].values)
    pred_rev = (1-TREND_BLEND)*pred_rev + TREND_BLEND*np.maximum(trend_rev, pred_rev)
    pred_cog = (1-TREND_BLEND)*pred_cog + TREND_BLEND*np.maximum(trend_cog, pred_cog)

    pred_rev = pred_rev * SHIFT_CONST
    pred_cog = pred_cog * SHIFT_CONST
    pred_cog = np.minimum(pred_cog, pred_rev * 0.99)

    # 12. Return
    result = pd.DataFrame({
        "date":    df.loc[test_mask, "date"].values,
        "revenue": np.round(pred_rev, 2),
        "cogs":    np.round(pred_cog, 2),
    })
    return result