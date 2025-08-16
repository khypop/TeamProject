# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image, ImageFile, UnidentifiedImageError
import io, torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
import os

# (옵션) HEIC 지원: 설치되어 있으면 자동 등록
try:
    import pillow_heif  # pip install pillow-heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

# 손상 이미지도 최대한 열도록
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
base_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(base_id).to(device).eval()
processor = CLIPProcessor.from_pretrained(base_id)

# 커스텀 가중치(있으면 로드). 실행 위치에 의존하지 않도록 파일 기준 경로 사용
ROOT = Path(__file__).resolve().parent
ckpt_path = Path(os.getenv("WEIGHTS_PATH", ROOT / "clip_coco.pth"))
try:
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt, strict=False)
        print(f"Custom weights loaded: {ckpt_path}")
    else:
        print(f"Use base weights (checkpoint not found at {ckpt_path})")
except Exception as e:
    print(f"Use base weights. Reason: {e}")

def _open_image_safe(raw: bytes) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(raw))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except (UnidentifiedImageError, OSError, ValueError):
        return None

@torch.no_grad()
def _text_feat(text: str):
    t = processor(text=[text], return_tensors="pt", padding=True)
    t = {k: v.to(device) for k, v in t.items()}
    return F.normalize(model.get_text_features(**t), dim=-1)  # [1,D]

@torch.no_grad()
def _image_feats(pils: List[Image.Image], bs: int = 16):
    feats = []
    for i in range(0, len(pils), bs):
        chunk = pils[i:i+bs]
        b = processor(images=chunk, return_tensors="pt")
        b = {k: v.to(device) for k, v in b.items()}
        f = model.get_image_features(**b)
        feats.append(F.normalize(f, dim=-1))
    return torch.cat(feats, dim=0) if feats else torch.empty(0, device=device)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/search")
async def search(
    text: str = Form(...),
    threshold: float = Form(0.3),
    files: List[UploadFile] = File(...)
):
    try:
        pils: List[Image.Image] = []
        names: List[str] = []

        for f in files:
            raw = await f.read()
            img = _open_image_safe(raw)
            if img is not None:
                pils.append(img)
                names.append(f.filename or "image")

        if not pils:
            return {"results": []}

        txt = _text_feat(text)            # [1,D]
        img = _image_feats(pils, bs=16)   # 배치 크기 조절 가능(8/16/32)
        if img.numel() == 0:
            return {"results": []}

        scores = (img @ txt.T).squeeze(-1).float().cpu().tolist()  # [-1,1]
        results = [
            {"name": n, "score": float(s)}
            for n, s in zip(names, scores) if s >= threshold
        ]
        results.sort(key=lambda x: x["score"], reverse=True)

        if device == "cuda":
            torch.cuda.empty_cache()

        return {"results": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "internal_error", "detail": str(e)})
