import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import os
import shutil

# 직접 학습시킨 모델 경로(원근)
CUSTOM_MODEL_PATH = "clip_coco.pth"

# CLIP 모델 및 전처리기 로드
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.load_state_dict(torch.load(CUSTOM_MODEL_PATH, map_location='cpu'))
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()

# 검색 결과 정렬 함수
def sort(results, sort_option):
    if sort_option == "정확도순(높은→낮은)":
        results.sort(key=lambda x: x[0], reverse=True)
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

# 이미지 저장 함수
def save_image(source_path, destination_path):
    try:
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.copy2(source_path, destination_path)
        return True, "이미지가 성공적으로 저장되었습니다!"
    except Exception as e:
        return False, f"저장 중 오류 발생: {str(e)}"

# 페이지 설정
st.set_page_config(page_title="CLIP 이미지 검색기", layout="wide")

# CSS 스타일링 - 저장 설정 박스 너비 제한
st.markdown("""
<style>
    .stExpander {
        max-width: 300px !important;
        width: 100% !important;
    }
    .stTextInput > div > div > input {
        max-width: 280px !important;
        width: 100% !important;
    }
    .stButton > button {
        max-width: 280px !important;
        width: 100% !important;
    }
    .stMarkdown {
        max-width: 280px !important;
        width: 100% !important;
    }
    .stCaption {
        max-width: 280px !important;
        width: 100% !important;
        word-wrap: break-word !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 텍스트/이미지로 내 폴더 이미지 검색 (CLIP 기반)")

# 세션 상태 초기화
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "search_performed" not in st.session_state:
    st.session_state.search_performed = False
if "save_messages" not in st.session_state:
    st.session_state.save_messages = []

# 사이드바 설정
with st.sidebar:
    sel_search_type = st.selectbox("검색할 방식", ("텍스트로 검색", "이미지로 검색"))
    sort_option = st.selectbox(
        "📂 결과 정렬 방식 선택",
        ("정확도순(높은→낮은)",
         "날짜순(최신)", "날짜순(오래된)",
         "파일명순(A~Z)", "파일명순(Z~A)",
         "파일크기순(큰→작은)", "파일크기순(작은→큰)")
    )

# 이미지 폴더 경로 입력
image_folder = st.text_input("🔧 이미지 폴더 경로를 입력하세요", "C:/Users/USER/Pictures/image")

# 검색 프롬프트 또는 이미지 업로드
if sel_search_type == "텍스트로 검색":
    prompt = st.text_input("💬 검색할 텍스트 프롬프트", "a photo of the sea")
else:
    query_image = st.file_uploader("🖼️ 검색에 사용할 이미지를 선택하세요", type=["jpg", "jpeg", "png", "webp"])
        if query_image is not None:
        st.image(query_image, caption="검색에 사용할 이미지", width=100)

# 오류 로그 초기화
if "error" not in st.session_state:
    st.session_state.error = []

# 저장 경로 기본 설정
st.subheader("💾 이미지 저장 설정")
default_save_folder = st.text_input("📁 기본 저장 폴더 경로", "C:/Users/USER/Pictures/saved_images")
st.info("💡 위 경로에 선택한 이미지들이 저장됩니다. 각 이미지별로 개별 경로도 설정할 수 있습니다.")

# 검색 시작 버튼
if st.button("🔎 검색 시작"):
    image_paths = []
    if not os.path.exists(image_folder):
        st.error("❌ 폴더 경로가 존재하지 않습니다.")
    else:
        # 재귀적으로 폴더 내 모든 이미지 파일 수집
        def scan_images(folder):
            try:
                folder_list = list(os.scandir(folder))
            except:
                return
            for dir in folder_list:
                if dir.is_file():
                    if dir.name.lower().endswith((".jpg", ".png", ".jpeg", ".webp")):
                        image_paths.append(dir.path)
                elif dir.is_dir():
                    scan_images(dir.path)

        scan_images(image_folder)

        if len(image_paths) == 0:
            st.warning("⚠️ 이미지가 없습니다.")
        else:
            with st.spinner("CLIP 모델로 검색 중..."):
                # 텍스트 또는 이미지 임베딩 생성
                if sel_search_type == "텍스트로 검색":
                    text_inputs = processor(text=[prompt], return_tensors="pt", padding=True)
                    with torch.no_grad():
                        text_features = model.get_text_features(**text_inputs)[0]
                else:
                    uploaded_image = Image.open(query_image).convert("RGB")
                    inputs = processor(images=uploaded_image, return_tensors="pt")
                    with torch.no_grad():
                        query_features = model.get_image_features(**inputs)[0]

                # 이미지 임베딩 및 유사도 계산
                results = []
                for path in image_paths:
                    try:
                        image = Image.open(path).convert("RGB")
                        inputs = processor(images=image, return_tensors="pt")
                        with torch.no_grad():
                            image_features = model.get_image_features(**inputs)[0]
                            if sel_search_type == "텍스트로 검색":
                                score = torch.nn.functional.cosine_similarity(text_features, image_features, dim=0)
                            else:
                                score = torch.nn.functional.cosine_similarity(query_features, image_features, dim=0)
                        
                        # 검색 방식에 따른 유사도 필터링
                        if sel_search_type == "텍스트로 검색":
                            # 텍스트 검색: 유사도 0.24 이상
                            if score.item() >= 0.24:
                                results.append((score.item(), path))
                        else:
                            # 이미지 검색: 유사도 0.8 이상
                            if score.item() >= 0.8:
                                results.append((score.item(), path))
                    except Exception as e:
                        st.session_state.error.append(f"{path} 처리 중 오류 발생: {e}")

                st.session_state.search_results = results
                st.session_state.search_performed = True
                st.rerun()

# 저장 메시지 표시
if st.session_state.save_messages:
    for msg in st.session_state.save_messages:
        if msg["type"] == "success":
            st.success(msg["message"])
        else:
            st.error(msg["message"])
    st.session_state.save_messages = []

# 검색 결과 표시
if st.session_state.search_performed and st.session_state.search_results:
    results = st.session_state.search_results
    MAX_HEIGHT = 200
    DEFAULT_COLS = 5
    st.subheader(f"📸 검색 결과 ({len(results)}개):")

    # 결과 정렬
    results.sort(key=lambda x: x[0], reverse=True)
    if sort_option != "정확도순(높은→낮은)":
        sort(results, sort_option)

    # 이미지 그리드 레이아웃
    i = 0
    while i < len(results):
        remaining = len(results) - i
        cols_per_row = min(DEFAULT_COLS, remaining)
        current_batch = results[i:i + cols_per_row]
        cols = st.columns(cols_per_row, gap="small")

        # 각 컬럼에 이미지와 저장 설정 표시
        for j, (score, path) in enumerate(current_batch):
            with cols[j]:
                # 이미지 표시 (고정 크기)
                st.image(path, caption=f"{os.path.basename(path)}\n유사도: {score:.4f}", width=240, use_container_width=False)

                # 저장 설정 UI
                with st.expander(f"💾 저장 설정", expanded=False):
                    # 컨테이너로 너비 제한
                    with st.container():
                        # 기본 저장 경로 표시
                        st.markdown("**📁 기본 경로:**")
                        st.caption(default_save_folder)
                        
                        # 구분선
                        st.divider()
                        
                        # 개별 저장 경로 설정
                        custom_save_folder = st.text_input(
                            "📂 저장 폴더",
                            value=default_save_folder,
                            key=f"custom_folder_{i}_{j}",
                            help="개별 저장 경로 설정",
                            max_chars=50
                        )
                        
                        # 파일명 설정
                        original_filename = os.path.basename(path)
                        name, ext = os.path.splitext(original_filename)
                        
                        # 파일명과 확장자를 한 줄에 표시
                        col_name, col_ext = st.columns([4, 1])
                        with col_name:
                            custom_filename = st.text_input(
                                "📝 파일명",
                                value=name,
                                key=f"custom_filename_{i}_{j}",
                                help="파일명 (확장자 제외)",
                                max_chars=30
                            )
                        with col_ext:
                            st.markdown(f"**{ext}**")
                            st.markdown("")  # 간격 조정
                        
                        # 최종 저장 경로 미리보기
                        final_filename = f"{custom_filename}{ext}" if custom_filename else original_filename
                        final_save_path = os.path.join(custom_save_folder, final_filename)
                        
                        st.markdown("**🎯 저장 경로:**")
                        # 경로가 길면 말줄임표로 표시
                        if len(final_save_path) > 40:
                            display_path = final_save_path[:37] + "..."
                        else:
                            display_path = final_save_path
                        st.caption(display_path)
                        
                        # 저장 버튼
                        if st.button("💾 저장하기", key=f"save_{i}_{j}", use_container_width=True):
                            success, message = save_image(path, final_save_path)
                            if success:
                                st.session_state.save_messages.append({"type": "success", "message": message})
                            else:
                                st.session_state.save_messages.append({"type": "error", "message": message})
                            st.rerun()
        
        i += cols_per_row

    # 새로 검색하기 버튼
    if st.button("🔄 새로 검색하기"):
        st.session_state.search_performed = False
        st.session_state.search_results = None
        st.rerun()

# 오류 로그 표시
if st.session_state.error:
    with st.expander("오류 로그"):
        for er in st.session_state.error:
            st.write(er)


