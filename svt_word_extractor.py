import os
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm
import shutil
import random

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
SVT_DIR = "data\svt"
IMG_DIR = SVT_DIR

TRAIN_XML = os.path.join(SVT_DIR, "train.xml")
TEST_XML = os.path.join(SVT_DIR, "test.xml")

OUT_DIR = "data\words"
TRAIN_OUT = os.path.join(OUT_DIR, "train")
VAL_OUT = os.path.join(OUT_DIR, "val")

VAL_SPLIT = 0.1      # 10% validation
TARGET_HEIGHT = 32    # CRNN expects height fixed at 32

# -------------------------------------------------------
# Utility functions
# -------------------------------------------------------

def ensure_clean_dir(path):
    """Deletes and recreates folder."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def resize_to_height(img, target_h):
    h, w = img.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    resized = cv2.resize(img, (new_w, target_h))
    return resized


def parse_xml(xml_path):
    """Extracts word boxes from SVT XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    samples = []  # list of (img_path, [(word, x1,y1,x2,y2), ...])

    for img_tag in root.findall("image"):
        img_name = img_tag.find("imageName").text
        img_path = os.path.join(IMG_DIR, img_name)

        rects = img_tag.find("taggedRectangles")
        words = []

        for rect in rects.findall("taggedRectangle"):
            x = int(rect.get("x"))
            y = int(rect.get("y"))
            w = int(rect.get("width"))
            h = int(rect.get("height"))
            word = rect.find("tag").text.strip().upper()

            words.append((word, x, y, x + w, y + h))

        samples.append((img_path, words))

    return samples

# -------------------------------------------------------
# Main extraction
# -------------------------------------------------------

def extract_words(xml_path, output_dir):
    samples = parse_xml(xml_path)
    counter = 0

    for img_path, words in tqdm(samples, desc=f"Extracting from {os.path.basename(xml_path)}"):
        img = cv2.imread(img_path)
        if img is None:
            print("Could not read:", img_path)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        image_id = os.path.splitext(os.path.basename(img_path))[0]

        for idx, (word, x1, y1, x2, y2) in enumerate(words):
            crop = gray[y1:y2, x1:x2]

            # skip tiny crops or empty
            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                continue

            # normalize to height = 32
            crop = resize_to_height(crop, TARGET_HEIGHT)

            # output filename
            out_name = f"{word}_{image_id}_{idx}.png"
            out_path = os.path.join(output_dir, out_name)

            # save
            cv2.imwrite(out_path, crop)
            counter += 1

    return counter

# -------------------------------------------------------
# Run extraction + split
# -------------------------------------------------------

def main():
    print("Cleaning output folders...")
    ensure_clean_dir(OUT_DIR)
    ensure_clean_dir(TRAIN_OUT)
    ensure_clean_dir(VAL_OUT)

    print("\nExtracting TRAIN words...")
    n1 = extract_words(TRAIN_XML, TRAIN_OUT)
    print(f" → {n1} word crops extracted from train.xml")

    print("\nExtracting TEST words...")
    n2 = extract_words(TEST_XML, TRAIN_OUT)
    print(f" → {n2} word crops extracted from test.xml")

    # -------------------------------
    # Create train/val split
    # -------------------------------
    all_files = os.listdir(TRAIN_OUT)
    random.shuffle(all_files)

    val_count = int(len(all_files) * VAL_SPLIT)
    val_files = all_files[:val_count]

    for f in val_files:
        shutil.move(os.path.join(TRAIN_OUT, f), os.path.join(VAL_OUT, f))

    print(f"\nFinal dataset:")
    print(f"  Train: {len(os.listdir(TRAIN_OUT))} images")
    print(f"  Val:   {len(os.listdir(VAL_OUT))} images")

if __name__ == "__main__":
    main()
