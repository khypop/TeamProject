# finetune_clip_from_csv.py
import os, csv, random
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModel
from PIL import Image
from tqdm import tqdm

# =========================
# CONFIG (여기만 여러분 경로/옵션으로 수정)
# =========================
CFG = {
    "IMG_DIR": r"E:\koreait\teamproject\winter_image",            # 이미지 루트 폴더
    "CSV_PATH": r"E:\koreait\teamproject\captions_winter.csv",    # filename,caption CSV
    "CKPT_PATH": r"E:\koreait\teamproject\clip_coco_trained.pt",  # 기존 체크포인트(없으면 None)
    "OUT_DIR":  r"E:\koreait\teamproject\outputs",                # 저장 폴더

    "BATCH_SIZE": 32,
    "EPOCHS": 10,
    "EMBED_DIM": 512,
    "MAX_LEN": 48,
    "VAL_SPLIT": 0.1,
    "NUM_WORKERS": 2,

    "FT_STRATEGY": "freeze_all",  # 1단계: 'freeze_all' → 이후 'tune_all'로 재실행 권장
    "LR_ENCODERS": 1e-5,          # 인코더 학습률(작게)
    "LR_HEAD": 5e-4,              # projection/FC 학습률(크게)
    "AUG": True,                  # 데이터 증강
    "AMP": True,                  # 혼합정밀도
    "SEED": 42,
}

# =========================
# Dataset
# =========================
class CsvPairDataset(Dataset):
    """
    CSV: filename,caption
    filename 은 IMG_DIR 기준 상대경로이거나, 파일명만 들어있다고 가정.
    """
    def __init__(self, img_dir: str, csv_path: str,
                 tokenizer_name="distilbert-base-uncased", max_len=48, aug=True):
        self.img_dir = img_dir
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.items: List[Tuple[str, str]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert "filename" in reader.fieldnames and "caption" in reader.fieldnames, \
                "CSV에 'filename'과 'caption' 컬럼이 있어야 합니다."
            for row in reader:
                fn = row["filename"].strip()
                cap = (row["caption"] or "").strip()
                if not fn or not cap:
                    continue
                self.items.append((fn, cap))

        if aug:
            self.tf = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel, caption = self.items[idx]
        img_path = os.path.join(self.img_dir, rel) if os.path.sep not in rel else rel
        if not os.path.exists(img_path):
            # filename만 들어있을 때 대비 (IMG_DIR + basename)
            img_path = os.path.join(self.img_dir, os.path.basename(rel))
        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)

        tok = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = tok["input_ids"].squeeze(0)
        attn_mask = tok["attention_mask"].squeeze(0)
        return img, input_ids, attn_mask

# =========================
# Model (Mini-CLIP)
# =========================
class MiniCLIP(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.image = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.image.fc = nn.Linear(self.image.fc.in_features, embed_dim)
        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)

    def forward(self, pixel_values, input_ids, attention_mask):
        img_emb = self.image(pixel_values)                       # (B, D)
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_cls = txt_out.last_hidden_state[:, 0, :]             # [CLS]
        txt_emb = self.text_proj(txt_cls)
        img_emb = F.normalize(img_emb, dim=-1)
        txt_emb = F.normalize(txt_emb, dim=-1)
        return img_emb, txt_emb

def clip_loss(i_emb, t_emb, temp=0.07):
    logits = (t_emb @ i_emb.T) / temp
    labels = torch.arange(i_emb.size(0), device=i_emb.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2

def set_finetune_strategy(model: MiniCLIP, strategy: str):
    # 모두 동결 후 필요한 부분만 학습
    for p in model.parameters():
        p.requires_grad = False

    if strategy == "freeze_all":
        for p in model.image.fc.parameters(): p.requires_grad = True
        for p in model.text_proj.parameters(): p.requires_grad = True
    elif strategy == "freeze_text":
        for p in model.image.parameters(): p.requires_grad = True
        for p in model.text_proj.parameters(): p.requires_grad = True
    elif strategy == "freeze_image":
        for p in model.text_encoder.parameters(): p.requires_grad = True
        for p in model.text_proj.parameters(): p.requires_grad = True
        for p in model.image.fc.parameters(): p.requires_grad = True
    elif strategy == "tune_all":
        for p in model.parameters(): p.requires_grad = True
    else:
        raise ValueError("FT_STRATEGY must be one of [freeze_all, freeze_text, freeze_image, tune_all]")

def build_optimizer(model: MiniCLIP, lr_enc=1e-5, lr_head=5e-4):
    enc_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in ["image.fc", "text_proj"]):
            head_params.append(p)
        else:
            enc_params.append(p)
    return torch.optim.AdamW([
        {"params": enc_params, "lr": lr_enc},
        {"params": head_params, "lr": lr_head},
    ])

def run_epoch(model, loader, optimizer, device, scaler=None, train=True):
    model.train(train)
    running = 0.0
    for imgs, ids, mask in tqdm(loader, disable=False):
        imgs, ids, mask = imgs.to(device), ids.to(device), mask.to(device)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            i_emb, t_emb = model(imgs, ids, mask)
            loss = clip_loss(i_emb, t_emb)
        if train:
            optimizer.zero_grad()
            if scaler is None:
                loss.backward(); optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
        running += loss.item()
    return running / max(1, len(loader))

def main():
    random.seed(CFG["SEED"]); torch.manual_seed(CFG["SEED"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CFG["OUT_DIR"], exist_ok=True)

    # Dataset & split
    full = CsvPairDataset(CFG["IMG_DIR"], CFG["CSV_PATH"],
                          max_len=CFG["MAX_LEN"], aug=CFG["AUG"])
    n_val = max(1, int(len(full) * CFG["VAL_SPLIT"]))
    n_train = len(full) - n_val
    train_set, val_set = random_split(full, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(CFG["SEED"]))

    train_loader = DataLoader(train_set, batch_size=CFG["BATCH_SIZE"], shuffle=True,
                              num_workers=CFG["NUM_WORKERS"], pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=CFG["BATCH_SIZE"], shuffle=False,
                              num_workers=CFG["NUM_WORKERS"], pin_memory=True)

    # Model
    model = MiniCLIP(embed_dim=CFG["EMBED_DIM"]).to(device)

    # (옵션) 기존 체크포인트 로드
    if CFG["CKPT_PATH"] and os.path.exists(CFG["CKPT_PATH"]):
        ckpt = torch.load(CFG["CKPT_PATH"], map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
        print(f"[INFO] Loaded checkpoint: {CFG['CKPT_PATH']}")

    # Finetuning strategy & optimizer
    set_finetune_strategy(model, CFG["FT_STRATEGY"])
    optimizer = build_optimizer(model, CFG["LR_ENCODERS"], CFG["LR_HEAD"])
    scaler = torch.cuda.amp.GradScaler() if (CFG["AMP"] and device.type == "cuda") else None

    best_val = 1e9
    for epoch in range(1, CFG["EPOCHS"] + 1):
        tr_loss = run_epoch(model, train_loader, optimizer, device, scaler, train=True)
        va_loss = run_epoch(model, val_loader,   optimizer, device, scaler=None, train=False)
        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")

        # save best
        if va_loss < best_val:
            best_val = va_loss
            best_path = os.path.join(CFG["OUT_DIR"], "clip_custom_best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ Saved: {best_path}")

    last_path = os.path.join(CFG["OUT_DIR"], "clip_custom_last.pt")
    torch.save(model.state_dict(), last_path)
    print(f"  ✅ Saved: {last_path}")

if __name__ == "__main__":
    main()
