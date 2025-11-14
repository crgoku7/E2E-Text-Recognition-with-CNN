#!/usr/bin/env python3
"""
paper_pipeline.py
Single-file implementation (models + unsupervised + train + inference demo)
Based on: "End-to-End Text Recognition with Convolutional Neural Networks" (Wang et al.)
Author: ChatGPT (adapted for Tushar Bhatt)
"""
import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import random
import math

# Torch + torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from scipy.ndimage import maximum_filter1d
from scipy.signal import find_peaks
import cv2
import matplotlib.pyplot as plt
import subprocess
import sys

# ----------------------------
# === CONFIG (edit as needed)
# ----------------------------
DATA_DIR = "data"
PROCESSED_DIR = "processed_data"
MODEL_DIR = "saved_models"
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans_filters.npy")

# Dataset-specific directories (expected to be created by the user)
CHARS74K_TRAIN = os.path.join(PROCESSED_DIR, "chars74k", "train")  # ImageFolder layout: <class>/<img>.png
DETECTOR_TRAIN = os.path.join(PROCESSED_DIR, "detector", "train")  # subfolders text/ non_text
SVT_DIR = os.path.join(DATA_DIR, "svt")                           # for demo (img + train.xml if available)

# Model / data hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 32
PATCH_SIZE = 8
DETECTOR_N1 = 96
DETECTOR_N2 = 256
RECOGNIZER_N1 = 115
RECOGNIZER_N2 = 720
CHAR_MAP = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CLASSES = len(CHAR_MAP)

NUM_PATCHES_FOR_KMEANS = 100000  # lower for quick experiments; increase for final training
KMEANS_BATCH = 1000

# Training hyperparams
BATCH_SIZE = 64
EPOCHS = 12
LR = 1e-3

# Inference hyperparams
INFERENCE_SCALES = [0.5, 0.75, 1.0, 1.25]
DETECTION_THRESHOLD = 0.5
ROW_NMS_DELTA = 16  # neighborhood half-width for 1D NMS (in heatmap pixels)
FINAL_IOU_NMS = 0.5

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ----------------------------
# === Model Definitions
# ----------------------------
class PaperActivation(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        # max(0, |x| - alpha)
        return F.relu(torch.abs(x) - self.alpha)

class PaperCNN(nn.Module):
    """
    Paper CNN:
    Input 1x32x32 -> Conv1 (8x8) -> 25x25 -> AvgPool5 -> 5x5 -> Conv2 (4x4) -> 2x2 -> FC
    """
    def __init__(self, n1_filters, n2_filters, num_classes, alpha=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, n1_filters, kernel_size=PATCH_SIZE, stride=1, padding=0, bias=False)
        self.act1 = PaperActivation(alpha=alpha)
        self.pool1 = nn.AvgPool2d(kernel_size=5, stride=5)

        self.conv2 = nn.Conv2d(n1_filters, n2_filters, kernel_size=4, stride=1, padding=0, bias=False)
        self.act2 = PaperActivation(alpha=alpha)
        self.fc = nn.Linear(n2_filters * 2 * 2, num_classes)

        # lightweight init for conv2 and fc
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='linear')

    def forward(self, x):
        # x: B x 1 x 32 x 32
        x = self.conv1(x)      # B x n1 x 25 x 25
        x = self.act1(x)
        x = self.pool1(x)      # B x n1 x 5 x 5
        x = self.conv2(x)      # B x n2 x 2 x 2
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def load_kmeans_filters(self, kmeans_path, load_n=None):
        if not os.path.exists(kmeans_path):
            raise FileNotFoundError(f"kmeans filters not found at {kmeans_path}")
        centers = np.load(kmeans_path)  # shape (K, 64)
        if load_n is None: load_n = centers.shape[0]
        n_max = self.conv1.weight.shape[0]
        if load_n > n_max:
            raise ValueError("Trying to load more filters than conv1 channels")
        filt = centers[:load_n].reshape(load_n, 1, PATCH_SIZE, PATCH_SIZE)
        with torch.no_grad():
            self.conv1.weight[:load_n].copy_(torch.tensor(filt, dtype=torch.float32))
        print(f"Loaded {load_n} filters into conv1 from {kmeans_path}")

# ----------------------------
# === Unsupervised filter learning (patch extraction, ZCA, KMeans)
# ----------------------------
def zca_whiten(X, eps=1e-5):
    # X shape (N, D)
    X = X.astype(np.float32)
    mu = np.mean(X, axis=0, keepdims=True)
    Xc = X - mu
    sigma = np.cov(Xc, rowvar=False)
    U, S, _ = np.linalg.svd(sigma, full_matrices=False)
    Z = U.dot(np.diag(1.0 / np.sqrt(S + eps))).dot(U.T)
    Xw = Xc.dot(Z.T)
    return Xw, mu, Z

def get_all_char_images_tensor(folder=CHARS74K_TRAIN, max_images=None):
    """
    Returns a torch.Tensor of grayscale images (N, 1, H, W) loaded from an ImageFolder-like structure
    (expects subfolders per character).
    Images are loaded and converted to float32 in [0,1] range.
    """
    imgs = []
    classes = sorted(os.listdir(folder)) if os.path.exists(folder) else []
    for cls in classes:
        cls_path = os.path.join(folder, cls)
        if not os.path.isdir(cls_path): continue
        for f in os.listdir(cls_path):
            p = os.path.join(cls_path, f)
            try:
                im = Image.open(p).convert("L")
                arr = np.array(im).astype(np.float32) / 255.0
                imgs.append(arr)
                if max_images and len(imgs) >= max_images: break
            except Exception:
                continue
        if max_images and len(imgs) >= max_images: break
    if len(imgs) == 0:
        return None
    # pad / resize small images to at least IMG_SIZE
    processed = []
    for a in imgs:
        h, w = a.shape
        if h < IMG_SIZE or w < IMG_SIZE:
            a = cv2.resize(a, (max(w, IMG_SIZE), max(h, IMG_SIZE)))
        processed.append(a)
    max_h = max(a.shape[0] for a in processed)
    max_w = max(a.shape[1] for a in processed)
    # pad to same size
    stacked = []
    for a in processed:
        pad_h = max_h - a.shape[0]
        pad_w = max_w - a.shape[1]
        stacked.append(np.pad(a, ((0,pad_h),(0,pad_w)), mode='constant', constant_values=0))
    arr = np.stack(stacked, axis=0)
    tensor = torch.tensor(arr[:, None, :, :], dtype=torch.float32)
    return tensor

def learn_kmeans_filters(save_path=KMEANS_PATH, n_patches=NUM_PATCHES_FOR_KMEANS, n_clusters=None):
    print("Starting unsupervised filter learning...")
    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for learn_kmeans_filters but failed to import. "
            "Please install/repair scikit-learn and its dependencies."
        ) from e
    images = get_all_char_images_tensor(CHARS74K_TRAIN, max_images=5000)  # load subset for speed
    if images is None:
        raise RuntimeError(f"No images found in {CHARS74K_TRAIN}. Run prepare_chars74k to create the dataset.")
    N, C, H, W = images.shape
    print(f"Loaded {N} images for patch extraction (shape {H}x{W})")

    psize = PATCH_SIZE
    patches = np.zeros((n_patches, psize * psize), dtype=np.float32)
    for i in tqdm(range(n_patches), desc="Sampling patches"):
        idx = random.randrange(0, N)
        img = images[idx].squeeze(0).numpy()
        h, w = img.shape
        y = random.randint(0, h - psize)
        x = random.randint(0, w - psize)
        p = img[y:y+psize, x:x+psize].flatten()
        patches[i] = p

    # normalize per patch
    patches -= patches.mean(axis=1, keepdims=True)
    patches_std = patches.std(axis=1, keepdims=True) + 1e-8
    patches /= patches_std

    # ZCA (use subset if huge)
    print("Computing ZCA whitening (this may be slow)...")
    p_sub = patches[:20000] if patches.shape[0] > 20000 else patches
    _, mu, Z = zca_whiten(p_sub)
    # whiten whole set by applying Z computed on subset (approx)
    patches_whiten = (patches - mu).dot(Z.T)

    n_clusters_final = n_clusters or max(DETECTOR_N1, RECOGNIZER_N1)
    print(f"Running MiniBatchKMeans with {n_clusters_final} clusters...")
    k = MiniBatchKMeans(n_clusters=n_clusters_final, batch_size=KMEANS_BATCH, n_init=10, random_state=0)
    k.fit(patches_whiten)
    centers = k.cluster_centers_.astype(np.float32)
    np.save(save_path, centers)
    print(f"KMeans centers saved to {save_path}")
    return save_path

# ----------------------------
# === Dataset helpers (ImageFolder usage)
# ----------------------------
def make_imagefolder_loader(root_dir, train=True, batch_size=BATCH_SIZE, shuffle=True):
    """
    Expects ImageFolder layout: root_dir/<class_name>/*.png
    Returns DataLoader with transforms that create 32x32 grayscale images normalized to [-1,1]
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"{root_dir} not found")
    transform_train = transforms.Compose([
        transforms.Resize((40,40)),
        transforms.RandomAffine(degrees=15, translate=(0.15,0.15), scale=(0.8,1.2), shear=10),
        transforms.CenterCrop((IMG_SIZE,IMG_SIZE)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    transform_test = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    transform = transform_train if train else transform_test
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    return loader

# ----------------------------
# === Training: squared-hinge (L2-SVM) loss
# ----------------------------
class SquaredHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, outputs, labels):
        # outputs: B x C (raw scores), labels: B
        target = F.one_hot(labels, num_classes=outputs.shape[1]).float()
        target = target * 2.0 - 1.0  # +1 for true, -1 for others
        margins = 1.0 - target * outputs
        losses = torch.clamp(margins, min=0.0) ** 2
        return losses.mean()

def train_model(model, train_loader, val_loader=None, epochs=EPOCHS, lr=LR, save_path=None, device=DEVICE):
    device = torch.device(device)
    model.to(device)
    # freeze conv1 as in paper
    for p in model.conv1.parameters():
        p.requires_grad = False
    params = list(model.conv2.parameters()) + list(model.fc.parameters())
    optimizer = optim.Adam(params, lr=lr)
    criterion = SquaredHingeLoss()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            total += x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{100*correct/total:.2f}%")
        if val_loader is not None:
            model.eval()
            vloss = 0.0; vtotal=0; vcorrect=0
            with torch.no_grad():
                for x,y in val_loader:
                    x = x.to(device); y = y.to(device)
                    out = model(x); loss = criterion(out,y)
                    vloss += loss.item() * x.size(0); vtotal += x.size(0)
                    vcorrect += (out.argmax(1) == y).sum().item()
            print(f"Val: loss={vloss/vtotal:.4f}, acc={100*vcorrect/vtotal:.2f}%")
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")

# ----------------------------
# === Utilities: IoU, NMS
# ----------------------------
def bbox_iou(boxA, boxB):
    # box: [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    inter = interW * interH
    areaA = (boxA[2]-boxA[0]+1)*(boxA[3]-boxA[1]+1)
    areaB = (boxB[2]-boxB[0]+1)*(boxB[3]-boxB[1]+1)
    return inter / float(areaA + areaB - inter + 1e-8)

def nms_boxes(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        rest = idxs[1:]
        to_keep = []
        for j in rest:
            if bbox_iou(boxes[i], boxes[j]) <= iou_threshold:
                to_keep.append(j)
        idxs = np.array(to_keep)
    return keep

# ----------------------------
# === Inference: heatmap, 1D NMS, box formation, recognition
# ----------------------------
def compute_detector_heatmap(detector, gray_img, device=DEVICE, batch_size=1024):
    """
    gray_img: np.ndarray HxW (0..255)
    Slides a 32x32 window with stride 1 and evaluates detector to produce heatmap H_map x W_map
    Returns heatmap as np.float32
    """
    detector.eval()
    H, W = gray_img.shape
    h_map = H - IMG_SIZE + 1
    w_map = W - IMG_SIZE + 1
    if h_map <= 0 or w_map <= 0:
        return np.empty((0,0), dtype=np.float32)
    # normalize to [-1,1] per training transform (ToTensor + Normalize((0.5,), (0.5,)))
    img = torch.tensor(gray_img.astype(np.float32) / 255.0, dtype=torch.float32, device=device)
    img = (img - 0.5) / 0.5
    # unfold
    img = img.unsqueeze(0).unsqueeze(0)  # 1x1xH x W
    patches = img.unfold(2, IMG_SIZE, 1).unfold(3, IMG_SIZE, 1)  # 1 x 1 x h_map x w_map x 32 x 32
    patches = patches.contiguous().view(-1, 1, IMG_SIZE, IMG_SIZE)
    heatmap = torch.zeros((h_map * w_map,), dtype=torch.float32, device=device)
    with torch.inference_mode():
        for start in range(0, patches.size(0), batch_size):
            end = min(start + batch_size, patches.size(0))
            batch = patches[start:end]
            out = detector(batch)  # (B, 2)
            probs = F.softmax(out, dim=1)[:, 1]  # probability of 'text' class
            heatmap[start:end] = probs
    heatmap = heatmap.view(h_map, w_map).cpu().numpy()
    return heatmap

def process_heatmaps_to_line_boxes(response_maps, threshold=DETECTION_THRESHOLD, delta=ROW_NMS_DELTA):
    """
    response_maps: dict scale -> heatmap (h_map x w_map)
    For each scale, do row-wise 1D local-max NMS and threshold; form contiguous groups -> boxes
    Then map boxes to original image coords and finally perform IoU NMS across scales.
    """
    all_boxes = []
    all_scores = []
    for scale, heatmap in response_maps.items():
        h_map, w_map = heatmap.shape
        if h_map == 0 or w_map == 0: continue
        # local 1D maximum filter along columns (axis=1)
        local_max = maximum_filter1d(heatmap, size=2*delta+1, axis=1, mode='constant')
        mask = (heatmap >= local_max) & (heatmap >= threshold)
        for r in range(h_map):
            cols = np.where(mask[r])[0]
            if cols.size == 0: continue
            # group contiguous columns
            groups = []
            cur = [cols[0]]
            for c in cols[1:]:
                if c - cur[-1] <= 1:
                    cur.append(c)
                else:
                    groups.append(cur); cur=[c]
            if cur: groups.append(cur)
            for g in groups:
                x_min = g[0]
                x_max = g[-1]
                # map to original image coordinates. Each column corresponds to window whose left = col (stride=1)
                left = int(round(x_min / scale))
                right = int(round((x_max + IMG_SIZE - 1) / scale))
                top = int(round(r / scale))
                bottom = int(round((r + IMG_SIZE - 1) / scale))
                if right <= left or bottom <= top:
                    continue
                score = float(heatmap[r, g].mean())
                all_boxes.append([left, top, right, bottom])
                all_scores.append(score)
    if not all_boxes:
        return []
    keep_idxs = nms_boxes(all_boxes, all_scores, iou_threshold=FINAL_IOU_NMS)
    final = [all_boxes[i] for i in keep_idxs]
    final_scores = [all_scores[i] for i in keep_idxs]
    return final, final_scores

def get_score_matrix_for_line(recognizer, line_img_gray, device=DEVICE):
    """
    line_img_gray: HxW (numpy) - we will resize to height=IMG_SIZE and slide horizontally to build C x N matrix of log-probs
    returns score_matrix (NUM_CLASSES x num_windows) of log probabilities (numpy)
    """
    h, w = line_img_gray.shape
    if h == 0 or w == 0:
        return np.array([])

    new_w = int(round(w * (IMG_SIZE / float(h))))
    if new_w < IMG_SIZE:
        return np.array([])
    resized = cv2.resize(line_img_gray, (new_w, IMG_SIZE))
    num_windows = resized.shape[1] - IMG_SIZE + 1
    if num_windows <= 0:
        return np.array([])
    score_matrix = np.full((NUM_CLASSES, num_windows), -np.inf, dtype=np.float32)

    # recognizer transform (match training)
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    recognizer.eval()
    with torch.inference_mode():
        for j in range(num_windows):
            patch = resized[:, j:j+IMG_SIZE]
            inp = trans(patch).unsqueeze(0).to(device)  # 1x1x32x32
            out = recognizer(inp)  # 1xC
            logp = F.log_softmax(out, dim=1).squeeze(0).cpu().numpy()
            score_matrix[:, j] = logp
    return score_matrix

def viterbi_align(score_matrix, word, char_map=CHAR_MAP):
    """
    Simple Viterbi-style alignment: choose for each char a column index monotonic increasing.
    We add small geometric penalties: prefer moderate spacings.
    score_matrix: C x N (log-probs)
    returns (best_score, alignment_indices) or (-inf, []) if impossible
    """
    if score_matrix.size == 0:
        return -np.inf, []
    N = score_matrix.shape[1]
    L = len(word)
    # map characters to indices robustly
    char_indices = []
    for ch in word:
        idx = char_map.find(ch)
        if idx == -1:
            idx = char_map.find(ch.lower())
        if idx == -1:
            idx = char_map.find(ch.upper())
        if idx == -1:
            return -np.inf, []
        char_indices.append(idx)
    # DP: dp[i,j] = best score aligning first i+1 chars with column j as the i-th char
    dp = np.full((L, N), -np.inf, dtype=np.float32)
    prev = np.zeros((L, N), dtype=np.int32)
    # init
    dp[0, :] = score_matrix[char_indices[0], :]
    for i in range(1, L):
        for j in range(i, N):  # at least i columns to place i-th char
            prev_scores = dp[i-1, :j]
            if prev_scores.size == 0:
                continue
            spacings = j - np.arange(prev_scores.shape[0])
            # penalty: prefer spacings between 2..20
            penalty = np.ones_like(spacings, dtype=float)
            penalty[(spacings < 2) | (spacings > 20)] = 0.1
            cand = prev_scores + np.log(penalty + 1e-9)
            max_idx = np.argmax(cand)
            dp[i, j] = cand[max_idx] + score_matrix[char_indices[i], j]
            prev[i, j] = max_idx
    best_idx = np.argmax(dp[-1, :])
    if dp[-1, best_idx] == -np.inf:
        return -np.inf, []
    alignment = [0] * L
    alignment[-1] = int(best_idx)
    for i in range(L-2, -1, -1):
        alignment[i] = int(prev[i+1, alignment[i+1]])
    return float(dp[-1, best_idx]), alignment

# ----------------------------
# === Demo and evaluation small helpers
# ----------------------------
def run_demo(detector, recognizer, image_path, lexicon=None, visualize=True):
    """
    Runs detection -> recognition on a single image
    lexicon: list of candidate words to match (if None, demonstrate raw top chars)
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    response_maps = {}
    for scale in INFERENCE_SCALES:
        h = int(gray.shape[0] * scale); w = int(gray.shape[1] * scale)
        if h < IMG_SIZE or w < IMG_SIZE: continue
        resized = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap = compute_detector_heatmap(detector, resized, device=DEVICE)
        if heatmap.size == 0: continue
        response_maps[scale] = heatmap

    boxes, scores = process_heatmaps_to_line_boxes(response_maps, threshold=DETECTION_THRESHOLD, delta=ROW_NMS_DELTA)
    print(f"Detected {len(boxes)} line boxes")

    results = []
    for box in boxes:
        x1, y1, x2, y2 = box
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0: continue
        score_matrix = get_score_matrix_for_line(recognizer, crop, device=DEVICE)
        if score_matrix.size == 0:
            continue
        # simple lexicon matching if lexicon provided: viterbi for each word
        best_word = None; best_score = -np.inf
        if lexicon:
            for w in lexicon:
                s, _ = viterbi_align(score_matrix, w)
                if s > best_score:
                    best_score = s; best_word = w
        else:
            # fallback: for each column choose argmax char and collapse to string (very naive)
            cols = score_matrix.argmax(axis=0)
            chars = [CHAR_MAP[c] for c in cols]
            # simple collapse of repeated characters
            collapsed = []
            for ch in chars:
                if not collapsed or ch != collapsed[-1]:
                    collapsed.append(ch)
            best_word = "".join(collapsed)
        results.append({'box': box, 'word': best_word, 'score': best_score})

    # visualize
    if visualize:
        plt.figure(figsize=(12,8)); plt.imshow(img_np[:,:,::-1])
        ax = plt.gca()
        for res in results:
            x1,y1,x2,y2 = res['box']
            rect = plt.Rectangle((x1,y1), x2-x1, y2-y1, edgecolor='lime', facecolor='none', linewidth=2)
            ax.add_patch(rect)
            text = f"{res['word']} ({res['score']:.2f})" if res['score'] != -np.inf else f"{res['word']}"
            ax.text(x1, max(y1-8,0), text, color='red', fontsize=12, backgroundcolor='white')
        plt.axis('off'); plt.show()
    return results

# ----------------------------
# === SVT XML PARSER + DATASET
# ----------------------------
import xml.etree.ElementTree as ET

class SVTSample:
    def __init__(self, img_path, lexicon, boxes, words):
        self.img_path = img_path      # absolute path to image
        self.lexicon = lexicon        # list[str]
        self.gt_boxes = boxes         # list of [x1,y1,x2,y2]
        self.gt_words = words         # list[str]

def load_svt_xml(xml_path, img_dir):
    """
    Parse SVT train.xml or test.xml
    Return: list[SVTSample]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    samples = []
    for img_node in root.findall("image"):
        name = img_node.find("imageName").text     # e.g. "img/18_01.jpg"
        img_abs = os.path.join(img_dir, os.path.basename(name))

        # lex = comma-separated string
        lex_raw = img_node.find("lex").text.strip()
        lexicon = [x.strip() for x in lex_raw.split(",") if x.strip()]

        boxes, words = [], []
        rects_node = img_node.find("taggedRectangles")
        for rect in rects_node.findall("taggedRectangle"):
            x = int(rect.get("x"))
            y = int(rect.get("y"))
            w = int(rect.get("width"))
            h = int(rect.get("height"))

            tag = rect.find("tag").text.strip()
            boxes.append([x, y, x + w, y + h])
            words.append(tag)

        samples.append(SVTSample(img_abs, lexicon, boxes, words))
    return samples


# ----------------------------
# === SVT EVALUATION
# ----------------------------
def match_boxes_iou(pred_boxes, gt_boxes, iou_thresh=0.5):
    """
    For each GT box, find best matching predicted box.
    Return: list of (pred_idx or None)
    """
    matches = []
    used = set()
    for i, gt in enumerate(gt_boxes):
        best_j = None
        best_iou = 0
        for j, pb in enumerate(pred_boxes):
            if j in used:
                continue
            iou = bbox_iou(gt, pb)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thresh:
            used.add(best_j)
            matches.append(best_j)
        else:
            matches.append(None)
    return matches

def evaluate_svt(detector, recognizer, svt_samples):
    """
    Computes:
       - Detection recall (box IoU >= 0.5)
       - Word recognition accuracy (lexicon constrained)
       - End-to-end accuracy (GT box correctly recognized)
    """
    total_gt = 0
    detected_correct = 0
    recog_correct = 0
    e2e_correct = 0

    for sample in tqdm(svt_samples, desc="Evaluating SVT"):
        img = Image.open(sample.img_path).convert("RGB")
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # ---- Step 1: multi-scale response maps ----
        response_maps = {}
        for scale in INFERENCE_SCALES:
            h = int(gray.shape[0] * scale)
            w = int(gray.shape[1] * scale)
            if h < IMG_SIZE or w < IMG_SIZE: continue
            resized = cv2.resize(gray, (w, h))
            heatmap = compute_detector_heatmap(detector, resized)
            if heatmap.size > 0:
                response_maps[scale] = heatmap

        # ---- Step 2: boxes ----
        pred_boxes, _ = process_heatmaps_to_line_boxes(response_maps)

        # ---- Step 3: match GT boxes by IoU ----
        matches = match_boxes_iou(pred_boxes, sample.gt_boxes, iou_thresh=0.5)

        # ---- Step 4: recognize each predicted line ----
        pred_words = []
        for box in pred_boxes:
            x1, y1, x2, y2 = box
            crop = gray[y1:y2, x1:x2]
            score_matrix = get_score_matrix_for_line(recognizer, crop)
            # lexicon-viterbi
            best_w, best_s = None, -np.inf
            for w in sample.lexicon:
                s, _ = viterbi_align(score_matrix, w)
                if s > best_s:
                    best_s, best_w = s, w
            pred_words.append(best_w)

        # ---- Step 5: compute metrics ----
        total_gt += len(sample.gt_words)

        for k, gt_word in enumerate(sample.gt_words):
            pred_idx = matches[k]
            if pred_idx is not None:
                detected_correct += 1
                # recognition correctness?
                if pred_words[pred_idx] == gt_word:
                    e2e_correct += 1

        # independent recognition accuracy
        for pw, gt in zip(pred_words[:len(sample.gt_words)], sample.gt_words):
            if pw == gt:
                recog_correct += 1

    det_recall = detected_correct / total_gt
    recog_acc = recog_correct / total_gt
    e2e_acc = e2e_correct / total_gt

    print("------ SVT Evaluation ------")
    print(f"Total GT words       : {total_gt}")
    print(f"Detection Recall     : {det_recall*100:.2f}%")
    print(f"Recognition Accuracy : {recog_acc*100:.2f}%")
    print(f"E2E Accuracy         : {e2e_acc*100:.2f}%")

    return det_recall, recog_acc, e2e_acc

def run_demo(detector, recognizer, xml_path, img_dir, image_index=0, visualize=True):
    """
    Demo using SVT dataset.
    Picks the image at `image_index` from the XML file,
    loads its lexicon and ground-truth boxes,
    runs detection + recognition,
    and visualizes the results.
    """
    # ---- Load the XML dataset ----
    samples = load_svt_xml(xml_path, img_dir)
    if image_index >= len(samples):
        print(f"ERROR: SVT index {image_index} out of range (max {len(samples)-1})")
        return

    sample = samples[image_index]
    print(f"Demo Image: {sample.img_path}")
    print(f"Lexicon size: {len(sample.lexicon)}")
    print(f"GT words: {sample.gt_words}")

    # ---- Load image ----
    img = Image.open(sample.img_path).convert("RGB")
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # ---- Multi-scale heatmaps ----
    response_maps = {}
    for scale in INFERENCE_SCALES:
        h = int(gray.shape[0] * scale)
        w = int(gray.shape[1] * scale)
        if h < IMG_SIZE or w < IMG_SIZE:
            continue
        resized = cv2.resize(gray, (w, h))
        heatmap = compute_detector_heatmap(detector, resized)
        if heatmap.size > 0:
            response_maps[scale] = heatmap

    # ---- Extract line boxes ----
    pred_boxes, pred_scores = process_heatmaps_to_line_boxes(
        response_maps,
        threshold=DETECTION_THRESHOLD,
        delta=ROW_NMS_DELTA
    )
    print(f"Detected {len(pred_boxes)} lines")

    # ---- Recognize each line ----
    pred_words = []
    for box in pred_boxes:
        x1, y1, x2, y2 = box
        crop = gray[y1:y2, x1:x2]
        score_matrix = get_score_matrix_for_line(recognizer, crop)

        if score_matrix.size == 0:
            pred_words.append("<EMPTY>")
            continue

        # lexicon-Viterbi
        best_word, best_score = None, -np.inf
        for w in sample.lexicon:
            s, _ = viterbi_align(score_matrix, w)
            if s > best_score:
                best_score, best_word = s, w

        pred_words.append(best_word if best_word else "<NONE>")

    # ---- Optional: GT IoU matching ----
    matches = match_boxes_iou(pred_boxes, sample.gt_boxes, iou_thresh=0.5)
    correct = 0
    for i, gt_w in enumerate(sample.gt_words):
        mj = matches[i]
        if mj is not None and pred_words[mj] == gt_w:
            correct += 1

    print(f"End-to-End correct words (IoU>=0.5): {correct}/{len(sample.gt_words)}")

    # ---- Visualization ----
    if visualize:
        plt.figure(figsize=(12,8))
        plt.imshow(img_np[:, :, ::-1])
        ax = plt.gca()

        # predicted boxes
        for (box, word) in zip(pred_boxes, pred_words):
            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                       fill=False, edgecolor='lime', linewidth=2))
            ax.text(x1, max(0, y1-10), word, fontsize=12,
                    color='red', backgroundcolor='white')

        # ground truth boxes (optional, draw them for debugging)
        for (gt_box, gt_word) in zip(sample.gt_boxes, sample.gt_words):
            x1, y1, x2, y2 = gt_box
            ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                       fill=False, edgecolor='yellow', linewidth=1))
            ax.text(x1, y2+5, f"GT:{gt_word}", color='yellow', fontsize=10)

        plt.axis("off")
        plt.show()

    return {
        "pred_boxes": pred_boxes,
        "pred_words": pred_words,
        "gt_boxes": sample.gt_boxes,
        "gt_words": sample.gt_words
    }


# ----------------------------
# === CLI
# ----------------------------
def main_cli():
    parser = argparse.ArgumentParser(description="Paper-style E2E text pipeline (single-file)")
    parser.add_argument("--learn-filters", action="store_true", help="Run unsupervised kmeans filter learning")
    parser.add_argument("--train-detector", action="store_true", help="Train detector CNN (requires processed detector dataset)")
    parser.add_argument("--train-recognizer", action="store_true", help="Train recognizer CNN (requires processed Chars74k)")
    parser.add_argument("--run-demo", type=str, help="Run demo on IMAGE_PATH with paper CNN pipeline (optionally provide --lexicon file path)")
    parser.add_argument("--lexicon", type=str, help="Path to newline-separated lexicon words for demo")
    parser.add_argument("--save-prefix", type=str, default="paper", help="prefix for saving models")
    parser.add_argument("--evaluate-svt", action="store_true",
                        help="Evaluate on SVT dataset using evaluate_svt.py")
    parser.add_argument("--svt-split", type=str, default="test",
                        help="SVT split: train or test (kept for backward compatibility, l)")


    parser.add_argument("--eval-single", type=str,
                        help="Run single-image evaluation using single_eval.py . Pass IMAGE_PATH.")

    args = parser.parse_args()

    if args.learn_filters:
        learn_kmeans_filters(save_path=KMEANS_PATH, n_patches=NUM_PATCHES_FOR_KMEANS)

    if args.train_detector:
        # expects DETECTOR_TRAIN with ImageFolder layout (text/ non_text)
        print("Preparing detector model")
        detector = PaperCNN(DETECTOR_N1, DETECTOR_N2, 2)
        if os.path.exists(KMEANS_PATH):
            detector.load_kmeans_filters(KMEANS_PATH, load_n=DETECTOR_N1)
        train_loader = make_imagefolder_loader(DETECTOR_TRAIN, train=True, batch_size=BATCH_SIZE)
        save_path = os.path.join(MODEL_DIR, f"{args.save_prefix}_detector.pth")
        train_model(detector, train_loader, epochs=EPOCHS, lr=LR, save_path=save_path, device=DEVICE)

    if args.train_recognizer:
        print("Preparing recognizer model")
        recognizer = PaperCNN(RECOGNIZER_N1, RECOGNIZER_N2, NUM_CLASSES)
        if os.path.exists(KMEANS_PATH):
            recognizer.load_kmeans_filters(KMEANS_PATH, load_n=RECOGNIZER_N1)
        train_loader = make_imagefolder_loader(CHARS74K_TRAIN, train=True, batch_size=BATCH_SIZE)
        save_path = os.path.join(MODEL_DIR, f"{args.save_prefix}_recognizer.pth")
        train_model(recognizer, train_loader, epochs=EPOCHS, lr=LR, save_path=save_path, device=DEVICE)

    if args.evaluate_svt:

        eval_script = os.path.join(os.path.dirname(__file__), "utils/evaluate_svt.py")
        if not os.path.exists(eval_script):
            print(f"ERROR: evaluate_svt.py not found at {eval_script}")
            return

        try:
            subprocess.run([sys.executable, eval_script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"evaluate_svt.py failed with code {e.returncode}")
        return


    if args.eval_single:
        single_eval_script = os.path.join(os.path.dirname(__file__), "utils/single_eval.py")
        if not os.path.exists(single_eval_script):
            print(f"ERROR: single_eval.py not found at {single_eval_script}")
            return
        img_path = args.eval_single

        cmd = [sys.executable, single_eval_script, "--image", img_path, "--visualize"]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"single_eval.py failed with code {e.returncode}")
        return


    if args.run_demo:
        # load models
        detector_path = os.path.join(MODEL_DIR, f"{args.save_prefix}_detector.pth")
        recognizer_path = os.path.join(MODEL_DIR, f"{args.save_prefix}_recognizer.pth")
        if not os.path.exists(detector_path) or not os.path.exists(recognizer_path):
            print("Trained model(s) not found. Train both detector and recognizer first.")
            return
        detector = PaperCNN(DETECTOR_N1, DETECTOR_N2, 2).to(DEVICE)
        recognizer = PaperCNN(RECOGNIZER_N1, RECOGNIZER_N2, NUM_CLASSES).to(DEVICE)
        detector.load_state_dict(torch.load(detector_path, map_location=DEVICE))
        recognizer.load_state_dict(torch.load(recognizer_path, map_location=DEVICE))
        lexicon = None
        if args.lexicon and os.path.exists(args.lexicon):
            with open(args.lexicon, 'r') as f:
                lexicon = [l.strip() for l in f.readlines() if l.strip()]
        run_demo(detector, recognizer, args.run_demo, lexicon=lexicon, visualize=True)

if __name__ == "__main__":
    main_cli()
