import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os
import requests
from streamlit_drawable_canvas import st_canvas

# 페이지 설정
st.set_page_config(
    page_title="숫자 인식 앱",
    page_icon="🔢",
    layout="wide"
)

# ONNX 모델 다운로드 함수
@st.cache_resource
def download_model():
    model_url = "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mnist/model/mnist-8.onnx"
    model_path = "mnist_model.onnx"
    
    if not os.path.exists(model_path):
        with st.spinner("MNIST 모델을 다운로드하는 중..."):
            response = requests.get(model_url)
            with open(model_path, 'wb') as f:
                f.write(response.content)
            st.success("모델 다운로드 완료!")
    
    return model_path

# ONNX 모델 로드
@st.cache_resource
def load_model():
    model_path = download_model()
    session = ort.InferenceSession(model_path)
    return session

# 이미지 전처리 함수
def preprocess_image(image_data):
    if image_data is None:
        return None
    
    # PIL Image로 변환
    img = Image.fromarray(image_data.astype('uint8'), 'RGBA')
    
    # 투명도 제거하고 흰색 배경으로 변경
    background = Image.new('RGB', img.size, (0, 0, 0))
    background.paste(img, mask=img.split()[0])  
    img_gray = background.convert('L')
    
    # 리사이즈, 정규화
    img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)
    img_normalized = img_array.astype(np.float32) / 255.0
    img_final = img_normalized.reshape(1, 1, 28, 28)
    return img_final


# 예측 함수
def predict_digit(model, image):
    if image is None:
        return None, None
    
    # 예측 실행
    input_name = model.get_inputs()[0].name
    result = model.run(None, {input_name: image})
    
    # 결과 처리
    confidence = result[0][0]
    predicted_digit = np.argmax(confidence)

    def softmax(x):
        e_x = np.exp(x - np.max(x))  
        return e_x / e_x.sum()
    
    probabilities = softmax(confidence)
    return predicted_digit, probabilities

# 앱 시작
st.title("손글씨 숫자 인식 앱")

# 모델 로드
try:
    model = load_model()
    st.success("MNIST 모델이 성공적으로 로드되었습니다!")
except Exception as e:
    st.error(f"모델 로드 실패: {str(e)}")
    st.stop()

# 3개 컬럼 구성
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.header("숫자를 그려보세요")
    
    # 캔버스 설정
    canvas_result = st_canvas(
        stroke_width=15,  # 선 굵기
        stroke_color="#000000",  # 검은색 선
        background_color="#FFFFFF",  # 흰색 배경
        width=250,
        height=250,
        drawing_mode="freedraw",
        key="canvas"
    )


with col2:
    st.header("전처리된 이미지")
    
    # 캔버스가 비어있지 않을 때만 전처리된 이미지 표시
    if canvas_result.image_data is not None :
        processed_image = preprocess_image(canvas_result.image_data)
        if processed_image is not None:
            processed_img_display = processed_image.reshape(28, 28)
            st.image(processed_img_display, caption="모델 입력 이미지 (28x28)", width=250)
        else:
            st.info("이미지를 처리하는 중...")
    else:
        st.info("숫자를 그려주세요") 

with col3:
    st.header("예측 결과")
    
    # 캔버스가 비어있지 않을 때만 예측 실행
    if canvas_result.image_data is not None :
        # 이미지 전처리
        processed_image = preprocess_image(canvas_result.image_data)
        
        if processed_image is not None:
            # 예측 실행
            predicted_digit, probabilities = predict_digit(model, processed_image)
            
            if predicted_digit is not None:
                # 결과 표시
                max_probabilities = np.max(probabilities)
                st.markdown(f"""예측 번호 : {predicted_digit}      
                                예측 확률 : {max_probabilities:.2%}""")
                
            else:
                st.error("예측에 실패했습니다.")
        else:
            st.info("이미지를 처리하는 중...")
    else:
        st.info("숫자를 그려주세요")

