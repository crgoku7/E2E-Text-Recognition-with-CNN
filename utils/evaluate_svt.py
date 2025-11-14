#!/usr/bin/env python3
"""
Evaluate detections+OCR on SVT-style ground truth.

Supported GT formats (per line):
- image_rel_path \t x1,y1,x2,y2 \t transcription
- image_name 8 x1 y1 x2 y2 x3 y3 x4 y4 transcription   (SVT quad format)

Example:
 
"""
import os
import argparse
import cv2
import math
from difflib import SequenceMatcher
import xml.etree.ElementTree as ET

from single_eval import load_yolo, detect_boxes, crop_image, ocr_easyocr, ocr_pytesseract


DETECTOR_WEIGHTS = "saved_models/best_fine_tuned.pt"
# SVT defaults
SVT_TEST_XML = os.path.join("data", "svt", "test.xml")
IMAGES_ROOT = os.path.join("data", "svt")
OCR_BACKEND = "easyocr"
OUTPUT_DIR = os.path.join("outputs", "eval_svt")

def parse_gt_line(line):
    # try TSV: img\tbbox\tword
    parts = line.strip().split("\t")
    if len(parts) == 3:
        img = parts[0]
        bbox_str = parts[1]
        word = parts[2].strip().strip('"')
        # bbox can be "x1,y1,x2,y2" or "x1 y1 x2 y2"
        if "," in bbox_str:
            vals = [float(x) for x in bbox_str.split(",")]
        else:
            vals = [float(x) for x in bbox_str.split()]
        if len(vals) >= 4:
            x1,y1,x2,y2 = vals[0],vals[1],vals[2],vals[3]
            return img, (int(round(x1)),int(round(y1)),int(round(x2)),int(round(y2))), word
    # try SVT quad format: img_name 8 x1 y1 x2 y2 x3 y3 x4 y4 word
    parts2 = line.strip().split()
    if len(parts2) >= 10:
        img = parts2[0]
        # second token sometimes '8'
        coord_start = 2 if parts2[1].isdigit() else 1
        coords = [float(x) for x in parts2[coord_start:coord_start+8]]
        xs = coords[0::2]; ys = coords[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        word = " ".join(parts2[coord_start+8:]).strip().strip('"')
        return img, (int(round(x1)),int(round(y1)),int(round(x2)),int(round(y2))), word
    raise ValueError("Unknown GT line format: " + line)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH
    boxAArea = (boxA[2]-boxA[0]+1) * (boxA[3]-boxA[1]+1)
    boxBArea = (boxB[2]-boxB[0]+1) * (boxB[3]-boxB[1]+1)
    if boxAArea <= 0 or boxBArea <= 0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)

def normalize_text(s):
    return "".join(ch for ch in s.strip().upper() if not ch.isspace())

def text_match_ratio(a, b):
    a = a.strip().upper(); b = b.strip().upper()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def evaluate(args):
    # parse SVT test.xml and build GT entries
    gt_entries = {}
    if not os.path.exists(SVT_TEST_XML):
        raise RuntimeError(f"SVT test xml not found at {SVT_TEST_XML}")
    tree = ET.parse(SVT_TEST_XML)
    root = tree.getroot()
    # XML structure: <tagset><image><imageName>img/xx.jpg</imageName><taggedRectangles><taggedRectangle x=".." y=".." width=".." height=".."><tag>WORD</tag></taggedRectangle>...</taggedRectangles></image>...</tagset>
    for img_el in root.findall('image'):
        name_el = img_el.find('imageName')
        if name_el is None:
            continue
        img_rel = name_el.text.strip()
        rects = img_el.find('taggedRectangles')
        if rects is None:
            continue
        for tr in rects.findall('taggedRectangle'):
            try:
                x = float(tr.get('x'))
                y = float(tr.get('y'))
                w = float(tr.get('width'))
                h = float(tr.get('height'))
            except Exception:
                continue
            x1 = int(round(x))
            y1 = int(round(y))
            x2 = int(round(x + w - 1))
            y2 = int(round(y + h - 1))
            tag_el = tr.find('tag')
            word = tag_el.text.strip() if tag_el is not None and tag_el.text else ""
            gt_entries.setdefault(img_rel, []).append({"box": (x1, y1, x2, y2), "word": word})


    detector = load_yolo(DETECTOR_WEIGHTS)
    # ensure output dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # init OCR reader
    reader = None
    ocr_choice = OCR_BACKEND if not hasattr(args, 'ocr') else args.ocr
    if ocr_choice == "easyocr":
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:
            use_gpu = False
        try:
            reader = __import__("easyocr").Reader(['en'], gpu=use_gpu)
        except Exception:
            reader = None

    total_gt = 0
    gt_matched = 0
    gt_correct_text = 0
    gt_matched_wrong_text = 0
    total_preds = 0
    tp_det = 0  # true positives for detection (matched preds)
    tp_end2end = 0  # true positives for end-to-end (matched + correct text)

    for img_rel, gts in gt_entries.items():
        img_path = os.path.join(IMAGES_ROOT, img_rel) if not os.path.isabs(img_rel) else img_rel
        if not os.path.exists(img_path):
            print("Missing image:", img_path)
            continue
        preds = detect_boxes(detector, img_path, conf_thres=args.conf)
        img_cv = cv2.imread(img_path)
        pred_list = []
        for (x1,y1,x2,y2,score,cls_id) in preds:
            crop = crop_image(img_cv, (x1,y1,x2,y2), pad=4)
            if crop.shape[0] < 12 or crop.shape[1] < 12:
                crop = cv2.resize(crop, (max(12,crop.shape[1]*2), max(12,crop.shape[0]*2)), interpolation=cv2.INTER_CUBIC)
            # Use initialized reader (easyocr) if available, else pytesseract
            if reader is not None:
                try:
                    text, conf = ocr_easyocr(reader, crop)
                except Exception:
                    text, conf = ocr_pytesseract(crop)
            else:
                text, conf = ocr_pytesseract(crop)
            pred_list.append({"box": (x1,y1,x2,y2), "text": text, "conf": conf, "score": score})
        # Perform greedy IoU matching between GTs and predictions so each pred matches at most one GT
        num_gts = len(gts)
        num_preds = len(pred_list)
        total_gt += num_gts
        total_preds += num_preds

        # build IoU list
        iou_pairs = []
        for gi, gt in enumerate(gts):
            for pi, p in enumerate(pred_list):
                iou_v = iou(gt["box"], p["box"])
                iou_pairs.append((iou_v, gi, pi))

        # sort by IoU descending and greedily assign
        iou_pairs.sort(reverse=True, key=lambda x: x[0])
        matched_g = set()
        matched_p = set()
        match_map_pred_to_gt = {}
        match_map_gt_to_pred = {}
        for iou_v, gi, pi in iou_pairs:
            if iou_v < args.iou:
                break
            if gi in matched_g or pi in matched_p:
                continue
            # match
            matched_g.add(gi)
            matched_p.add(pi)
            match_map_pred_to_gt[pi] = gi
            match_map_gt_to_pred[gi] = pi
            gt_matched += 1
            gt_word = gts[gi]["word"]
            pred_text = pred_list[pi]["text"]
            ratio = text_match_ratio(gt_word, pred_text)
            if ratio >= args.text_match_thresh:
                gt_correct_text += 1
                tp_end2end += 1
            else:
                gt_matched_wrong_text += 1

        # detection true positives: number of matched predictions
        tp_det += len(matched_p)

        # Save annotated image showing GT and predicted boxes and texts
        try:
            vis = img_cv.copy()
            # draw GT boxes (blue)
            for gi, gt in enumerate(gts):
                x1,y1,x2,y2 = gt["box"]
                gt_text = gt["word"]
                color = (255, 0, 0)  # blue for GT
                thickness = 2
                cv2.rectangle(vis, (x1,y1), (x2,y2), color, thickness)
                cv2.putText(vis, f"GT:{gt_text}", (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # draw predicted boxes
            for pi, p in enumerate(pred_list):
                x1,y1,x2,y2 = p["box"]
                pred_text = p["text"]
                score = p.get("score", 0.0)
                if pi in matched_p:
                    # matched prediction
                    gi = match_map_pred_to_gt.get(pi)
                    # check text correctness
                    gt_text = gts[gi]["word"] if gi is not None else ""
                    if text_match_ratio(gt_text, pred_text) >= args.text_match_thresh:
                        color = (0, 255, 0)  # green = matched and correct
                    else:
                        color = (0, 165, 255)  # orange = matched but wrong text
                else:
                    color = (0, 0, 255)  # red = unmatched prediction
                cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
                txt = f"P:{pred_text} ({score:.2f})"
                cv2.putText(vis, txt, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            out_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
            cv2.imwrite(out_path, vis)
        except Exception:
            # don't crash evaluation if saving visualization fails
            pass

    # metrics
    recall = gt_matched / total_gt if total_gt else 0.0
    word_acc = gt_correct_text / total_gt if total_gt else 0.0
    matched_and_wrong_pct = gt_matched_wrong_text / total_gt if total_gt else 0.0

    precision_det = tp_det / total_preds if total_preds else 0.0
    recall_det = tp_det / total_gt if total_gt else 0.0
    f1_det = (2 * precision_det * recall_det / (precision_det + recall_det)) if (precision_det + recall_det) else 0.0

    precision_e2e = tp_end2end / total_preds if total_preds else 0.0
    recall_e2e = tp_end2end / total_gt if total_gt else 0.0
    f1_e2e = (2 * precision_e2e * recall_e2e / (precision_e2e + recall_e2e)) if (precision_e2e + recall_e2e) else 0.0

    print("Total GT words:", total_gt)
    print(f"Total predicted boxes: {total_preds}")
    print(f"Detection - TP: {tp_det}, Precision: {precision_det:.3f}, Recall: {recall_det:.3f}, F1: {f1_det:.3f}")
    print(f"GT boxes found (IoU>={args.iou}): {gt_matched} ({recall:.3f})")
    print(f"Recognition (on GT matches) - correct: {gt_correct_text}, accuracy (over GT): {word_acc:.3f}")
    print(f"End-to-end - TP(correct text): {tp_end2end}, Precision: {precision_e2e:.3f}, Recall: {recall_e2e:.3f}, F1: {f1_e2e:.3f}")
    print(f"GT matched but wrong text: {gt_matched_wrong_text} ({matched_and_wrong_pct:.3f})")

if __name__ == "__main__":
    # No CLI arguments: evaluate on SVT test.xml with hardcoded model and paths
    class _Args:
        conf = 0.25
        iou = 0.5
        text_match_thresh = 0.8

    evaluate(_Args)