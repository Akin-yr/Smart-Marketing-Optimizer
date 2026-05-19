import os
import io
import pandas as pd
from configparser import ConfigParser
from sqlalchemy import create_engine
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FOLDER_ID = '1uSiEtiLEBM0IgmK866e53T4gX46u_c_k' # ID folder chứa file của bạn

# Chạy lệnh xin quyền cấp đọc file (readonly) từ Google Drive API
def get_gdrive_service():
    """Xác thực và kết nối Google Drive API"""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

# Lấy thông tin cấu hình từ file config để truy cập vào dtb
def load_db_config(filename='config/database.ini', section='postgresql'):
    """Đọc cấu hình kết nối database từ file .ini"""
    parser = ConfigParser()
    parser.read(filename)
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception(f'Section {section} not found in the {filename} file')
    return db

def main():
    # 1. Khởi tạo Drive Service
    service = get_gdrive_service()
    
    # 2. Lấy tất cả các file trong folder (không quan tâm là Excel hay CSV)
    query = f"'{FOLDER_ID}' in parents and trashed = false"
    # Lấy thêm trường mimeType để phân biệt loại file
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    files = results.get('files', [])
    
    if not files:
        print("Không tìm thấy file nào trong folder chỉ định.")
        return
    
    print(f"Tìm thấy {len(files)} file trong folder. Bắt đầu tiến trình.")

    # Code tránh file bị trùng tên nhưng khác domain, có thể làm thủ công trên google drive bằng cách xóa các files duplicated. Tuy nhiên để quản trị dtb một cách linh hoạt hơn ta vẫn khuyến khích việc lọc trùng file bằng script tránh việc nếu số lượng files quá lớn sẽ dẫn đến việc tốn tài nguyên vào việc xử duplicates.
    
    # Kết nối Postgres chuẩn bị sẵn
    db_params = load_db_config()
    conn_str = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
    engine = create_engine(conn_str)
    
    for f in files:
        file_name = f['name']
        file_id = f['id']
        mime_type = f['mimeType']
        
        is_excel = mime_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' or file_name.endswith('.xlsx')
        is_csv = mime_type == 'text/csv' or file_name.endswith('.csv')
        
        if not (is_excel or is_csv):
            continue
            
        print(f"Đang xử lý file: {file_name}")
        request = service.files().get_media(fileId=file_id)
        file_stream = io.BytesIO()
        downloader = MediaIoBaseDownload(file_stream, request)
        
        done = False
        while not done:
            _, done = downloader.next_chunk()
            
        file_stream.seek(0)
        
        try:
            if is_excel:
                df_temp = pd.read_excel(file_stream, sheet_name=0)
            elif is_csv:
                df_temp = pd.read_csv(file_stream, encoding='utf-8')
            
            # Transform cho từng file
            # Chuẩn hóa tên cột chữ thường, xóa khoảng trắng
            df_temp.columns = [col.lower().strip() for col in df_temp.columns]
            # Phòng hờ trong cùng 1 file có cột bị trùng tên
            df_temp = df_temp.loc[:, ~df_temp.columns.duplicated()]
            
            # Loading stage
            # Ví dụ file là "Marketing Data 2026.xlsx" -> tên bảng là "marketing_data_2026"
            raw_table_name = os.path.splitext(file_name)[0] # Bỏ đuôi .xlsx/.csv
            clean_table_name = raw_table_name.lower().strip().replace(" ", "_") # Chuyển thành chữ thường, thay khoảng trắng bằng dấu gạch dưới
            
            # Đẩy thẳng file này thành 1 bảng riêng biệt
            df_temp.to_sql(clean_table_name, engine, if_exists='replace', index=False)
            print(f"Đã nạp thành công vào bảng: '{clean_table_name}'")
            
        except Exception as e:
            print(f"Lỗi khi xử lý file {file_name}: {e}")
            continue

    print("\nToàn bộ các file đã được rải đều thành các bảng riêng biệt trong Postgres")

if __name__ == '__main__':
    main()