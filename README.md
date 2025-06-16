# ✍️ Draw2Digit: 손글씨 숫자 인식 앱

Streamlit 기반의 웹 애플리케이션으로, 사용자가 직접 그린 손글씨 숫자를 ONNX 모델(MNIST)을 통해 실시간으로 예측합니다.  
모델은 ONNX 형식의 사전 학습된 MNIST 모델을 사용하며, 캔버스를 통해 숫자를 자유롭게 그릴 수 있습니다.

---

## 🚀 주요 기능

- ✏️ 캔버스를 이용한 숫자 입력 (Streamlit-Drawable-Canvas)
- 🧠 ONNX 기반 MNIST 모델 추론
- 📊 예측 결과 및 softmax 확률 시각화 (Plotly)
- 💾 입력한 숫자 이미지 및 예측 결과 자동 저장 기능

---

## 🐳 Docker 사용 방법

### 1. 이미지 다운로드

```bash
docker pull jju54/draw2digit-app
```

### 2. 컨테이너 실행

```bash
docker run -p 8501:8501 jju54/draw2digit-app
```

### 3. 앱 접속
브라우저에서 다음 주소로 접속합니다:
👉 http://localhost:8501

## 📁 프로젝트 구조

```bash
draw2digit2/
├── app.py               # Streamlit 앱 실행 파일
├── saved_digits/        # 저장된 이미지 및 예측 결과
├── requirements.txt     # 필요한 파이썬 라이브러리
└── Dockerfile           # 도커 이미지 빌드 설정 파일
```


## 📦 requirements.txt 주요 패키지
- streamlit
- numpy
- Pillow
- onnxruntime
- streamlit-drawable-canvas
- plotly
- requests