#!/usr/bin/env python3
"""

Usage:

Requirements:
  pip install ultralytics pillow opencv-python numpy matplotlib
  For EasyOCR: pip install easyocr
  For pytesseract: pip install pytesseract  and install Tesseract (OS package)
"""

import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from ultralytics import YOLO

# OCR backends (optional imports)
try:
    import easyocr
except Exception:
    easyocr = None

try:
    import pytesseract
except Exception:
    pytesseract = None

def load_yolo(weights):
    model = YOLO(weights)
    return model

def detect_boxes(model, image_path, conf_thres=0.25):
    """
    Run inference and return list of boxes: [ (x1,y1,x2,y2, score, cls_id) ... ]
    Coordinates are ints in image pixel space.
    """
    results = model(image_path)[0]  # take first (Ultralytics API)
    boxes = []
    if results.boxes is None:
        return boxes

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf[0].item()) if hasattr(box, "conf") else float(box[4].item())
        cls_id = int(box.cls[0].item()) if hasattr(box, "cls") else -1
        if score < conf_thres:
            continue
        boxes.append((int(x1), int(y1), int(x2), int(y2), score, cls_id))
    return boxes

def crop_image(img_cv, box, pad=2):
    x1,y1,x2,y2 = box
    H,W = img_cv.shape[:2]
    x1 = max(0, x1-pad); y1 = max(0, y1-pad)
    x2 = min(W-1, x2+pad); y2 = min(H-1, y2+pad)
    crop = img_cv[y1:y2+1, x1:x2+1]
    return crop

# -------------------------
# OCR wrappers
# -------------------------
def ocr_easyocr(reader, crop_bgr):
    """
    crop_bgr: OpenCV BGR image
    returns: best_text (str), confidence (float)
    """
    # EasyOCR expects RGB
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    # reader.readtext returns list of (bbox, text, confidence)
    try:
        res = reader.readtext(crop_rgb, detail=1, paragraph=False)
    except Exception as e:
        # sometimes easyocr errors for tiny images
        return "", 0.0
    if not res:
        return "", 0.0
    # Choose highest confidence entry
    best = max(res, key=lambda r: r[2])
    return best[1], float(best[2])

def ocr_pytesseract(crop_bgr, config="--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    # convert to RGB PIL
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop_rgb)
    try:
        text = pytesseract.image_to_string(pil, config=config)
    except Exception:
        return "", 0.0
    text = text.strip()
    # pytesseract doesn't return confidence reliably for single words; return dummy conf
    conf = 0.0
    return text, conf

# -------------------------
# Visualization helpers
# -------------------------
def draw_boxes_and_text(img_bgr, detections, out_path=None):
    """
    detections: list of dicts: {box:(x1,y1,x2,y2), text:str, conf:float}
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1,y1,x2,y2 = det["box"]
        text = det.get("text","")
        conf = det.get("conf",0.0)
        # box
        draw.rectangle([x1,y1,x2,y2], outline="lime", width=2)
        # text background
        label = f"{text} {conf:.2f}"
        # PIL.ImageDraw may not have textsize in some versions; use textbbox or font.getsize as fallback
        try:
            bbox = draw.textbbox((x1, y1), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            try:
                tw, th = font.getsize(label)
            except Exception:
                tw, th = (len(label) * 6, 12)
        draw.rectangle([x1, max(0,y1-th-4), x1+tw+4, y1], fill="yellow")
        draw.text((x1+2, max(0,y1-th-4)+1), label, fill="black", font=font)

    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    if out_path:
        cv2.imwrite(out_path, out)
    return out

# -------------------------
# Main pipeline
# -------------------------
def process_image(detector_weights, image_path , conf_thres=0.25, visualize=False, out_path=None):
    """
    Runs model -> recognizer on single image
    Returns list of detections with text and confidence
    """
    recognizor_backend = 'easyocr'
    if recognizor_backend == "easyocr" and easyocr is None:
        raise RuntimeError("easyocr not installed. pip install easyocr")

    if recognizor_backend == "pytesseract" and pytesseract is None:
        raise RuntimeError("pytesseract not installed. pip install pytesseract and install tesseract.")

    detector = load_yolo(detector_weights)
    boxes = detect_boxes(detector, image_path, conf_thres=conf_thres)
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise RuntimeError("Could not load image: " + image_path)

    detections = []

    # init easyocr reader once
    reader = None
    if recognizor_backend == "easyocr":
        reader = easyocr.Reader(['en'], gpu=True if __import__('torch').cuda.is_available() else False)

    for (x1,y1,x2,y2,score,cls_id) in boxes:
        crop = crop_image(img_cv, (x1,y1,x2,y2), pad=4)
        # if too small, upsample a bit
        if crop.shape[0] < 12 or crop.shape[1] < 12:
            crop = cv2.resize(crop, (max(12,crop.shape[1]*2), max(12,crop.shape[0]*2)), interpolation=cv2.INTER_CUBIC)

        if recognizor_backend == "easyocr":
            text, conf = ocr_easyocr(reader, crop)
        else:
            text, conf = ocr_pytesseract(crop)

        detections.append({
            "box": (x1,y1,x2,y2),
            "det_conf": score,
            "text": text,
            "conf": conf,
            "cls": cls_id
        })

    # visualize
    vis_img = None
    if visualize:
        vis_img = draw_boxes_and_text(img_cv, detections, out_path)
    return detections, vis_img

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--weights", type=str,default="saved_models/best_fine_tuned.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--out", type=str, default="out.png")
    args = parser.parse_args()

    dets, vis = process_image(args.weights, args.image, conf_thres=args.conf, visualize=args.visualize, out_path=args.out if args.visualize else None)

    # print results
    for i, d in enumerate(dets):
        print(f"{i}: box={d['box']}, det_conf={d['det_conf']:.3f}, ocr_text='{d['text']}', ocr_conf={d['conf']:.3f}")

    if vis is not None:
        print("Wrote visualization to", args.out)
