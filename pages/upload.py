import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
# BlipProcessor은 이미지 전처리, 토큰화 / BlipForConditionalGeneration은 텍스트 생성 모델

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base") # 이 Salesforce/blip-image-captioning-base를 model에 저장함.

st.title("이미지 업로드 후 텍스트 추출")
image_upload = st.file_uploader("upload", type=["jpg", "png"]) # 이미지 파일만을 업로드해야 함.

with st.sidebar:
  st.page_link("clip_image_search_app.py", label = "Home", icon = "🏠") # 사이드바에 이 페이지로 연결되는 링크를 추가.

if image_upload is None:
  st.write("이미지 파일을 올려주세요.")
else:
  image = Image.open(image_upload) # 이미지 객체를 image에 저장.
  bliprocessor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base") # 이 모델에서 blipprocessor를 받아서 쓸거야. blipprocessor은 이미지 전처리와 토큰화, 디코드를 진행함.
  token = bliprocessor(images = image, return_tensors="pt") # 이미지의 전처리된 결과를 BlipProcessor의 키워드 인자는 항상 복수형임. 이미지 객체로 넘겨줘야 함. 이미지는 이미 숫자로 구성되었기 때문에 텐서(자료구조)로 만들어줌.
  come = model.generate(**token) # token은 딕셔너리 구조로 저장되기 때문에 ** 연산자를 통해 풀어줘야 함. generate는 문장을 생성해주는 거임. 인코딩 과정을 적어준 거지만 실제로는 모델 안에서 인코딩, 임베딩이 진행됨.
  text = bliprocessor.decode(come[0], skip_special_tokens=True) # 최종 시퀀스 텐서가 나오고 처음 리스트 인덱스(시퀀스 텐서)만 뽑으면 됨. 가장 가능성 높은 문장을 맨 앞에 놓음.
  st.subheader("\n\n[ {0} ]".format(text))
  st.image(image) # 이미지 객체 출력.
