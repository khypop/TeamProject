import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel

# 데이터셋 디렉토리 및 캡션 파일 경로 설정
ROOT_DIR = r"C:\Users\com\.cache\kagglehub\datasets\awsaf49\coco-2017-dataset\versions\2\coco2017"
IMAGE_DIR = os.path.join(ROOT_DIR, "train2017")
ANNOTATION_PATH = os.path.join(ROOT_DIR, "annotations", "captions_train2017.json")


# COCO 데이터셋 클래스 정의
class COCODataset(Dataset):
    def __init__(self, image_dir, annotation_path, processor):
        self.image_dir = image_dir
        self.processor = processor

        # 캡션 파일 로드
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 이미지 ID -> 파일명 매핑
        id2filename = {img['id']: img['file_name'] for img in data['images']}
        self.samples = [(id2filename[ann['image_id']], ann['caption']) for ann in data['annotations']]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 이미지와 캡션 로딩
        image_name, caption = self.samples[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        # 이미지와 텍스트 전처리
        inputs = self.processor(text=caption, images=image, return_tensors="pt", padding=True)
        return {k: v.squeeze(0) for k, v in inputs.items()}

# 배치 데이터 패딩 및 정리
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch], batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in batch], batch_first=True, padding_value=0
    )

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

# 학습 메인 함수
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 보안 문제 회피용 옵션
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        use_safetensors=True
    ).to(device)

    # 전처리기 로드
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 커스텀 COCO 데이터셋 준비
    dataset = COCODataset(IMAGE_DIR, ANNOTATION_PATH, processor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, collate_fn=collate_fn)

    # 옵티마이저 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    epochs = 7


    # 학습 루프
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in loop:
            # 입력을 GPU로 이동
            inputs = {k: v.to(device) for k, v in batch.items()}

            # 순전파 및 손실 계산
            outputs = model(**inputs, return_loss=True)
            loss = outputs.loss

            # 역전파 및 가중치 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # 에폭당 평균 손실 출력
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | 평균 Loss: {avg_loss:.4f}")

    # 학습 완료 후 모델 저장
    torch.save(model.state_dict(), "clip_coco.pth")
    print("모델 저장 완료: clip_coco.pth")


# Windows 환경에서 다중 프로세스 오류 방지용
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
