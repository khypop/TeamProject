import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import os

# 모델 및 전처리기 로드
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()

# 페이지 설정
st.set_page_config(page_title="CLIP 이미지 검색기", layout="wide")
st.title("🔍 텍스트로 내 폴더 이미지 검색 (CLIP 기반)")

# 이미지 폴더 선언
image_folder = st.text_input("🔧 이미지 폴더 경로를 입력하세요", "C:/Users/USER/Pictures/image")

# 프롬프트 입력
prompt = st.text_input("💬 검색할 텍스트 프롬프트", "a photo of the sea")

image = []

# 검색 버튼
if st.button("🔎 검색 시작"): # 검색 시작 버튼을 누르면
    if not os.path.exists(image_folder): # 선택한 폴더에 특정 파일이나 디렉토리가 존재하지 않으면. 즉, 입력한 디렉터리 그 자체임.
        st.error("❌ 폴더 경로가 존재하지 않습니다.") # error 추가
    else:
        for root, dirs, files in os.walk(image_folder): # 입력한 폴더에서 하위폴더까지의 이미지 파일을 찾음.
            image_paths = [os.path.join(root, i) for i in files if i.endswith((".jpg", ".png", ".jpeg"))]
        if len(image_paths) == 0:
            st.warning("⚠️ 이미지가 없습니다.")
        else:
            with st.spinner("CLIP 모델로 검색 중..."):
                # 프롬프트 임베딩
                text_inputs = processor(text=[prompt], return_tensors="pt", padding=True)
                with torch.no_grad():
                    text_features = model.get_text_features(**text_inputs)[0]

                # 이미지 임베딩 및 유사도 계산
                results = []
                for path in image_paths:
                    try:
                        image = Image.open(path).convert("RGB")
                        inputs = processor(images=image, return_tensors="pt")
                        with torch.no_grad():
                            image_features = model.get_image_features(**inputs)[0]
                            score = torch.nn.functional.cosine_similarity(text_features, image_features, dim=0)
                        results.append((score.item(), path))
                    except Exception as e:
                        st.write(f"{path} 처리 중 오류 발생: {e}")

                # 결과 정렬 및 표시
                results.sort(reverse=True)
                top_k = min(5, len(results))
                st.subheader(f"📸 상위 {top_k}개 결과:")
                cols = st.columns(top_k)
                for i in range(top_k):
                    score, path = results[i]
                    with cols[i]:
                        st.image(path, caption=f"{os.path.basename(path)}\n유사도: {score*100:.2f}%\n\n{os.path.join(image_folder, root)}", use_column_width=True)