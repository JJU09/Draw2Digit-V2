# 베이스 이미지 (Streamlit + Python)
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 파일 복사
COPY requirements.txt .
COPY app.py .
COPY saved_digits ./saved_digits

# 라이브러리 설치
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# 포트 오픈 (Streamlit 기본: 8501)
EXPOSE 8501

# 실행 명령
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]