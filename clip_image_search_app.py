import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import os
import shutil

# 직접 학습시킨 CLIP 모델 경로 (원근 모델)
CUSTOM_MODEL_PATH = "clip_coco.pth"

# CLIP 모델 및 전처리기 로드 함수 (캐싱해서 매번 재로딩 방지)
@st.cache_resource
def load_model():
    # 기본 CLIP 모델 로드
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # 커스텀 학습된 모델 파라미터 로드
    model.load_state_dict(torch.load(CUSTOM_MODEL_PATH, map_location='cpu'))
    
    # 이미지/텍스트 전처리기 로드
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# 모델과 프로세서 로드
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
        # 저장 경로가 없으면 생성
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        # 이미지 복사
        shutil.copy2(source_path, destination_path)
        return True, "이미지가 성공적으로 저장되었습니다!"
    except Exception as e:
        return False, f"저장 중 오류 발생: {str(e)}"

# 페이지 기본 설정
st.set_page_config(page_title="CLIP 이미지 검색기", layout="wide")

# CSS 스타일링 (레이아웃 조정)
st.markdown("""
<style>
    .stExpander { max-width: 300px !important; width: 100% !important; }
    .stTextInput > div > div > input { max-width: 280px !important; width: 100% !important; }
    .stButton > button { max-width: 280px !important; width: 100% !important; }
    .stMarkdown { max-width: 280px !important; width: 100% !important; }
    .stCaption { max-width: 280px !important; width: 100% !important; word-wrap: break-word !important; }
</style>
""", unsafe_allow_html=True)

# 앱 타이틀
st.title("🔍 텍스트/이미지로 내 폴더 이미지 검색 (CLIP 기반)")

# 세션 상태 초기화
if "search_results" not in st.session_state: st.session_state.search_results = None
if "search_performed" not in st.session_state: st.session_state.search_performed = False
if "save_messages" not in st.session_state: st.session_state.save_messages = []
if "text_history" not in st.session_state: st.session_state.text_history = []
if "image_history" not in st.session_state: st.session_state.image_history = []
if "error" not in st.session_state: st.session_state.error = []

# 사이드바 UI
with st.sidebar:
    # 검색 방식 선택 (텍스트 / 이미지)
    sel_search_type = st.selectbox("검색할 방식", ("텍스트로 검색", "이미지로 검색"))
    
    # 결과 정렬 옵션
    sort_option = st.selectbox(
        "📂 결과 정렬 방식 선택",
        ("정확도순(높은→낮은)",
         "날짜순(최신)", "날짜순(오래된)",
         "파일명순(A~Z)", "파일명순(Z~A)",
         "파일크기순(큰→작은)", "파일크기순(작은→큰)")
    )

    # 검색 히스토리 표시
    st.subheader("🕘 검색 히스토리")
    if sel_search_type == "텍스트로 검색":
        if st.session_state.text_history:
            for idx, hist in enumerate(reversed(st.session_state.text_history)):
                hist_idx = len(st.session_state.text_history) - idx - 1
                label = hist["prompt"]
                if st.button(f"{label}", key=f"sidebar_text_{hist_idx}"):
                    st.session_state.search_results = hist["results"]
                    st.session_state.search_performed = True
                    st.rerun()
        else:
            st.info("텍스트 검색 기록이 없습니다.")
    else:
        if st.session_state.image_history:
            for idx, hist in enumerate(reversed(st.session_state.image_history)):
                hist_idx = len(st.session_state.image_history) - idx - 1
                label = hist["image"].name if hist["image"] else "이미지"
                st.image(hist["image"], width=80)
                if st.button(f"{label}", key=f"sidebar_image_{hist_idx}"):
                    st.session_state.search_results = hist["results"]
                    st.session_state.search_performed = True
                    st.rerun()
        else:
            st.info("이미지 검색 기록이 없습니다.")

# 검색할 이미지 폴더 경로 입력
image_folder = st.text_input("🔧 이미지 폴더 경로를 입력하세요", "E:/Teamproject/image-up")

# 검색 프롬프트 또는 이미지 업로드
if sel_search_type == "텍스트로 검색":
    prompt = st.text_input("💬 검색할 텍스트 프롬프트", "a photo of the sea")
else:
    query_image = st.file_uploader("🖼️ 검색에 사용할 이미지를 선택하세요", type=["jpg", "jpeg", "png", "webp"])
    if query_image is not None:
        st.image(query_image, caption="검색에 사용할 이미지", width=150)

# 이미지 저장 기본 경로 설정
st.subheader("💾 이미지 저장 설정")
default_save_folder = st.text_input("📁 기본 저장 폴더 경로", "C:/Users/USER/Pictures/saved_images")
st.info("💡 위 경로에 선택한 이미지들이 저장됩니다. 각 이미지별로 개별 경로도 설정할 수 있습니다.")

# 검색 시작 버튼
if st.button("🔎 검색 시작"):
    image_paths = []
    if not os.path.exists(image_folder):
        st.error("❌ 폴더 경로가 존재하지 않습니다.")
    else:
        # 폴더 내 이미지 파일 재귀 탐색
        def scan_images(folder):
            try:
                folder_list = list(os.scandir(folder))
            except:
                return
            for dir in folder_list:
                if dir.is_file() and dir.name.lower().endswith((".jpg", ".png", ".jpeg", ".webp")):
                    image_paths.append(dir.path)
                elif dir.is_dir():
                    scan_images(dir.path)

        scan_images(image_folder)

        if len(image_paths) == 0:
            st.warning("⚠️ 이미지가 없습니다.")
        else:
            with st.spinner("CLIP 모델로 검색 중..."):
                # 텍스트 임베딩 또는 이미지 임베딩 생성
                if sel_search_type == "텍스트로 검색":
                    text_inputs = processor(text=[prompt], return_tensors="pt", padding=True)
                    with torch.no_grad():
                        text_features = model.get_text_features(**text_inputs)[0]
                else:
                    uploaded_image = Image.open(query_image).convert("RGB")
                    inputs = processor(images=uploaded_image, return_tensors="pt")
                    with torch.no_grad():
                        query_features = model.get_image_features(**inputs)[0]

                # 각 이미지와의 유사도 계산
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
                        
                        # 유사도 기준 필터링
                        if sel_search_type == "텍스트로 검색":
                            if score.item() >= 0.24:
                                results.append((score.item(), path))
                        else:
                            if score.item() >= 0.8:
                                results.append((score.item(), path))
                    except Exception as e:
                        st.session_state.error.append(f"{path} 처리 중 오류 발생: {e}")

                # 검색 결과 세션에 저장
                st.session_state.search_results = results
                st.session_state.search_performed = True

                # 검색 히스토리 저장 (최대 3개)
                history_item = {
                    "type": sel_search_type,
                    "prompt": prompt if sel_search_type == "텍스트로 검색" else None,
                    "image": query_image if sel_search_type == "이미지로 검색" else None,
                    "results": results
                }
                if sel_search_type == "텍스트로 검색":
                    st.session_state.text_history.append(history_item)
                    st.session_state.text_history = st.session_state.text_history[-3:]
                else:
                    st.session_state.image_history.append(history_item)
                    st.session_state.image_history = st.session_state.image_history[-3:]

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

    # 결과 정렬 (사용자 선택 반영)
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

        for j, (score, path) in enumerate(current_batch):
            with cols[j]:
                # 이미지 출력
                st.image(path, caption=f"{os.path.basename(path)}\n유사도: {score:.4f}", width=240, use_container_width=False)
                
                # 저장 옵션 확장
                with st.expander(f"💾 저장", expanded=False):
                    original_filename = os.path.basename(path)
                    name, ext = os.path.splitext(original_filename)
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
                        st.markdown("")  

                    # 저장 버튼
                    if st.button("💾 저장하기", key=f"save_{i}_{j}", use_container_width=True):
                        final_filename = f"{custom_filename}{ext}"
                        destination_path = os.path.join(default_save_folder, final_filename)
                        success, message = save_image(path, destination_path)
                        if success:
                            st.session_state.save_messages.append({"type": "success", "message": message})
                        else:
                            st.session_state.save_messages.append({"type": "error", "message": message})
                        st.rerun()
        
        i += cols_per_row

    # 새 검색 버튼
    if st.button("🔄 새로 검색하기"):
        st.session_state.search_performed = False
        st.session_state.search_results = None
        st.rerun()

# 오류 로그 표시
if st.session_state.error:
    with st.expander("오류 로그"):
        for er in st.session_state.error:
            st.write(er)
