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
with st.sidebar:
    num_of_output = st.number_input("출력할 사진의 갯수", 5)
    sort_option = st.selectbox(
        "📂 결과 정렬 방식 선택",
        ("정확도순(높은→낮은)", "정확도순(낮은→높은)",
         "날짜순(최신)", "날짜순(오래된)",
         "파일명순(A~Z)", "파일명순(Z~A)",
         "파일크기순(큰→작은)", "파일크기순(작은→큰)")
    )
    page2 = st.page_link("pages/upload.py", label="Image Upload", icon="⬆️") # pages 폴더도 다운 받아야 함.
# 이미지 폴더 선택
image_folder = st.text_input("🔧 이미지 폴더 경로를 입력하세요", "C:/Users/USER/Pictures/image")

# 프롬프트 입력
prompt = st.text_input("💬 검색할 텍스트 프롬프트", "a photo of the sea")

# 검색 버튼
if st.button("🔎 검색 시작"):
    if not os.path.exists(image_folder):
        st.error("❌ 폴더 경로가 존재하지 않습니다.")
    else:
        for root, dirs, files in os.walk(image_folder):
            image_paths = [os.path.join(root, f) for f in os.listdir(files) if f.endswith((".jpg", ".png", ".jpeg", ".webp"))]
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
                # ✅ 정렬 적용
                if sort_option == "정확도순(높은→낮은)":
                    results.sort(key=lambda x: x[0], reverse=True)
                elif sort_option == "정확도순(낮은→높은)":
                    results.sort(key=lambda x: x[0])
                elif sort_option == "날짜순(최신)":
                    results.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
                elif sort_option == "날짜순(오래된)":
                    results.sort(key=lambda x: os.path.getmtime(x[1]))
                elif sort_option == "파일명순(A~Z)":
                    results.sort(key=lambda x: os.path.basename(x[1]).lower())
                elif sort_option == "파일명순(Z~A)":
                    results.sort(key=lambda x: os.path.basename(x[1]).lower(), reverse=True)
                elif sort_option == "파일크기순(큰→작은)":
                    results.sort(key=lambda x: os.path.getsize(x[1]), reverse=True)
                elif sort_option == "파일크기순(작은→큰)":
                    results.sort(key=lambda x: os.path.getsize(x[1]))

                # 결과 정렬 및 표시
                results.sort(reverse=True)
                top_k = min(int(num_of_output), len(results))
                st.subheader(f"📸 상위 {top_k}개 결과:")
                cols = st.columns(top_k)
                for i in range(top_k):
                    score, path = results[i]
                    with cols[i]:
                        st.image(path, caption=f"{os.path.basename(path)}\n유사도: {score:.4f}", use_container_width=True)
