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

#정렬함수==============================================================================
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
#=======================================================================================

# 페이지 이름, 타이틀 설정
st.set_page_config(page_title="CLIP 이미지 검색기", layout="wide")
st.title("🔍 텍스트로 내 폴더 이미지 검색 (CLIP 기반)")

#사이드바 설정
with st.sidebar:
    num_of_output = st.number_input("출력할 사진의 갯수", 5)
    #----------------------selectbox 추가-------------------------------------------
    sel_search_type = st.selectbox("검색할 방식",("텍스트로 검색", "이미지로 검색"))
    sort_option = st.selectbox(
        "📂 결과 정렬 방식 선택",
        ("정확도순(높은→낮은)",
         "날짜순(최신)", "날짜순(오래된)",
         "파일명순(A~Z)", "파일명순(Z~A)",
         "파일크기순(큰→작은)", "파일크기순(작은→큰)")
    )
    #page2 = st.page_link("pages/upload.py", label="Image Upload", icon="⬆️") # pages 폴더도 다운 받아야 함.

# 텍스트로 이미지를 검색할 폴더 선택
image_folder = st.text_input("🔧 이미지 폴더 경로를 입력하세요", "C:/Users/USER/Pictures/image")

# 프롬프트 입력
if sel_search_type == "텍스트로 검색" :
    prompt = st.text_input("💬 검색할 텍스트 프롬프트", "a photo of the sea")
# 이미지로 검색 ----------------------------추가부분------------------------------------
else:
    query_image = st.file_uploader("🖼️ 검색에 사용할 이미지를 드래그 또는 선택하세요", type=["jpg", "jpeg", "png", "webp"])

image_paths = []
error = []
# 검색 버튼
if st.button("🔎 검색 시작"):
    if not os.path.exists(image_folder):
        st.error("❌ 폴더 경로가 존재하지 않습니다.")
    else:
        def lower(folder):
            try:
                folder = list(os.scandir(folder))
            except PermissionError:
                return
            except Exception:
                return
            try:
                for dir in folder:
                    if dir.is_file():
                        if dir.name.lower().endswith((".jpg", ".png", ".jpeg", ".webp"))
                            image_paths.append(dir.path)
                    elif dir.is_dir():
                        lower(dir.path)
            except PermissionError:
                pass
            except Exception as e:
                error.append(f"{path} 처리 중 오류 발생: {e}")
        image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".jpg", ".png", ".jpeg", ".webp"))]
        if len(image_paths) == 0:
            st.warning("⚠️ 이미지가 없습니다.")
        else:
            with st.spinner("CLIP 모델로 검색 중..."):
                # 프롬프트 임베딩
                if sel_search_type == "텍스트로 검색":
                    text_inputs = processor(text=[prompt], return_tensors="pt", padding=True)
                    with torch.no_grad():
                        text_features = model.get_text_features(**text_inputs)[0]

                # 이미지 임베딩 ----------------------------------------------추가-------------------------------------------
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
                            if sel_search_type == "텍스트로 검색": #텍스트로 검색시 유사도
                                score = torch.nn.functional.cosine_similarity(text_features, image_features, dim=0)

                            else:                                 #이미지로 검색시 유사도----------------------------------------추가--------------------------------------------
                                score = torch.nn.functional.cosine_similarity(query_features, image_features, dim=0)
                        results.append((score.item(), path))
                    except Exception as e:
                        error.append(f"{path} 처리 중 오류 발생: {e}")



                #이미지 표시 크기
                i = 0
                MAX_HEIGHT = 180
                DEFAULT_COLS = 5
                REDUCED_COLS = 4

                # 지정한 수와 파일 내 이미지 비교해서 적은 값 할당
                top_k = min(int(num_of_output), len(results))
                st.subheader(f"📸 상위 {top_k}개 결과:")
                
                #정확도 순으로 정렬 후 아래값 제거
                results.sort(key=lambda x: x[0], reverse=True)
                results = results[0:top_k]
                if sort_option != "정확도순(높은→낮은)":
                    sort(results, sort_option)

                while i < top_k:
                    remaining = top_k - i
                    batch = results[i:i + DEFAULT_COLS]        #다음 줄에 표현할 후보 이미지들 
                    cols_per_row = min(DEFAULT_COLS, remaining)#한 줄에 넣을 이미지 기본값 = 5

                    # 줄에 큰 이미지 있으면 4개로 줄이기
                    for _, path in batch:
                        try:
                            with Image.open(path) as img:
                                _, height = img.size
                                if height > MAX_HEIGHT:  #만약 사이즈가 크다면 한줄에 4개만 출력되도록 제한 
                                    cols_per_row = min(REDUCED_COLS, remaining)
                                    break
                        except:
                            cols_per_row = min(REDUCED_COLS, remaining)

                    current_batch = results[i:i + cols_per_row]
                    cols = st.columns(len(current_batch))  # 여기서 딱 맞게 column 생성


                    #결과 표시하기
                    for j, (score, path) in enumerate(current_batch):
                        with cols[j]:
                            st.image(path, caption=f"{os.path.basename(path)}\n유사도: {score:.4f}")
                    i += cols_per_row

                    if error:
                        with st.expander("오류 로그")
                            for er in error:
                                st.write(er)
    
