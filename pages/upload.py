import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
# BlipProcessor은 이미지 전처리, 토큰화 / BlipForConditionalGeneration은 텍스트 생성 모델

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

st.title("이미지 업로드 후 텍스트 추출")
image_upload = st.file_uploader("upload", type=["jpg", "png"])

with st.sidebar:
  st.page_link("main.py", label = "Home", icon = "🏠")

if image_upload is None:
  st.write("이미지 파일을 올려주세요.")
else:
  image = Image.open(image_upload)
  bliprocessor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base") # 이 모델에서 블립프로세서를 받아서 쓸거야
  token = bliprocessor(images = image, return_tensors="pt") # BlipProcessor의 키워드 인자는 항상 복수형임. 이미지 객체로 넘겨줘야 함.
  come = model.generate(**token)
  text = bliprocessor.decode(come[0], skip_special_tokens=True) # 최종 시퀀스 텐서가 나오고 처음 리스트 인덱스(시퀀스 텐서)만 뽑으면 됨. 가장 가능성 높은 문장을 맨 앞에
  st.subheader("\n\n[ {0} ]".format(text))
  st.image(image)