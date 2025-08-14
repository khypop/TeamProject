# json_to_csv.py
import os, json, csv
from urllib.parse import urlparse, unquote

# ========== CONFIG: 여기를 여러분 환경에 맞게 수정 ==========
IMG_DIR   = r"E:\koreait\teamproject\haerin_image"        # 실제 이미지가 있는 루트 폴더
JSON_PATH = r"E:\koreait\teamproject\newjeans-haerin.json"   # Label Studio에서 Export한 JSON (또는 COCO JSON)
OUT_CSV   = r"E:\koreait\teamproject\captions_haerin.csv" # 저장될 CSV 경로
# 파일명만 쓸지(True) / JSON의 상대경로를 유지할지(False)
USE_BASENAME_ONLY = True
# ==========================================================

def norm(p: str) -> str:
    return p.replace("\\", "/")

def pick_name_from_url(u: str) -> str:
    """file:///… 또는 http(s)://… → 파일명만 추출"""
    p = urlparse(u).path
    return os.path.basename(unquote(p))

def file_exists(img_dir: str, rel: str) -> bool:
    return os.path.exists(os.path.join(img_dir, rel))

assert os.path.isfile(JSON_PATH), f"JSON 파일을 찾을 수 없습니다: {JSON_PATH}"
assert os.path.isdir(IMG_DIR), f"이미지 폴더를 찾을 수 없습니다: {IMG_DIR}"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
missing = []

# ------- CASE 1: COCO 캡션 포맷 -------
if isinstance(data, dict) and "images" in data and "annotations" in data:
    id2file = {}
    for img in data["images"]:
        fn = norm(img.get("file_name", ""))
        if not fn:
            continue
        if USE_BASENAME_ONLY:
            fn = os.path.basename(fn)
        id2file[img["id"]] = fn

    for ann in data["annotations"]:
        img_id = ann.get("image_id")
        cap = (ann.get("caption") or "").strip()
        fn = id2file.get(img_id)
        if not fn or not cap:
            continue
        if not file_exists(IMG_DIR, fn):
            missing.append(fn)
        rows.append([fn, cap])

# ------- CASE 2: Label Studio Common JSON (tasks list) -------
elif isinstance(data, list):
    # 보통: [{ "data":{"image": ...}, "annotations":[{"result":[{"value":{"text":[...]}}]}] }, ...]
    for task in data:
        src = task.get("data", {}).get("image") or task.get("data", {}).get("img")
        if not src:
            continue

        # 경로 결정
        if USE_BASENAME_ONLY:
            fn = pick_name_from_url(src)
        else:
            # 필요시 JSON에 상대경로가 있다면 아래에서 적절히 가공
            fn = norm(unquote(urlparse(src).path).lstrip("/"))
            if "/" not in fn:  # URL이거나 절대경로였다면 파일명만
                fn = os.path.basename(fn)

        # 캡션(복수) 수집
        texts = []
        for ann in task.get("annotations", []) or task.get("completions", []):
            for r in ann.get("result", []):
                val = r.get("value", {})
                if isinstance(val.get("text"), list):
                    for t in val["text"]:
                        if isinstance(t, str) and t.strip():
                            texts.append(t.strip())

        if not texts:
            continue

        if not file_exists(IMG_DIR, fn):
            missing.append(fn)

        for cap in texts:
            rows.append([fn, cap])
else:
    raise ValueError("지원하지 않는 JSON 구조입니다. (COCO 또는 Label Studio Tasks JSON이어야 합니다)")

# CSV 저장
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["filename", "caption"])
    w.writerows(rows)

print(f"✅ CSV 저장 완료: {OUT_CSV} (rows={len(rows)})")
if missing:
    print(f"⚠️ 실제 폴더에서 찾을 수 없는 파일 {len(missing)}개 (상위 10개 예시):")
    for p in missing[:10]:
        print(" -", os.path.join(IMG_DIR, p))
else:
    print("✅ 모든 항목이 IMG_DIR에서 확인되었습니다.")
