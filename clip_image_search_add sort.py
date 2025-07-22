import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import os
from datetime import datetime

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

# ✅ 사이드바에 설정 메뉴 추가
st.sidebar.header("⚙️ 설정")
image_folder = st.sidebar.text_input("📁 이미지 폴더 경로", "C:/Users/USER/Pictures/image")

sort_option = st.sidebar.selectbox(
    "📂 이미지 정렬 방식",
    ("날짜순(최신)", "날짜순(오래된)", "파일명순(A~Z)", "파일명순(Z~A)", "파일크기순(큰→작은)", "파일크기순(작은→큰)")
)

# 프롬프트 입력 (메인 화면)
prompt = st.text_input("💬 검색할 텍스트 프롬프트 입력")

# 이미지 검색 버튼
if st.button("검색"):
    if os.path.exists(image_folder):
        image_files = [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # ✅ 정렬 기능 적용
        if sort_option == "날짜순(최신)":
            image_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        elif sort_option == "날짜순(오래된)":
            image_files.sort(key=lambda x: os.path.getmtime(x))
        elif sort_option == "파일명순(A~Z)":
            image_files.sort()
        elif sort_option == "파일명순(Z~A)":
            image_files.sort(reverse=True)
        elif sort_option == "파일크기순(큰→작은)":
            image_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
        elif sort_option == "파일크기순(작은→큰)":
            image_files.sort(key=lambda x: os.path.getsize(x))

        # 검색된 이미지 출력
        for image_file in image_files:
            img = Image.open(image_file)
            st.image(
                img,
                caption=f"{os.path.basename(image_file)} - 수정일: {datetime.fromtimestamp(os.path.getmtime(image_file)).strftime('%Y-%m-%d %H:%M:%S')}",
                use_column_width=True
            )
    else:
        st.error("❌ 지정한 경로가 존재하지 않습니다.")
