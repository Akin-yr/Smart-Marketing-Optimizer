import os.path
import io
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload # Thêm thư viện này để tải file

# Đổi SCOPE thành drive.readonly để có quyền tải file Excel
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# ID của FILE EXCEL thực tế trong folder của bạn (lấy từ URL của file excel đó)
EXCEL_FILE_ID = '1oWYYUm2jxkdA4RbwaQ_b6cpxuGN0JYM7' 

def main():
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

    # Gọi Drive API thay vì Sheets API
    drive_service = build('drive', 'v3', credentials=creds)

    # Tiến hành tải file Excel dưới dạng Binary Stream
    request = drive_service.files().get_media(fileId=EXCEL_FILE_ID)
    file_stream = io.BytesIO()
    downloader = MediaIoBaseDownload(file_stream, request)
    
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    
    # Đưa dữ liệu binary vào Pandas đọc trực tiếp
    file_stream.seek(0)
    df = pd.read_excel(file_stream, sheet_name=0)
    
    print("Đọc dữ liệu thành công! Đây là vài dòng đầu:")
    print(df.head())

if __name__ == '__main__':
    main()