#!/usr/bin/env python3
"""
===================================================================================================================
Author: Montadhar Guesmi
Date: 04/02/2025
Institution: TU Dresden, Institut für Verfahrenstechnik und Umwelttechnik,
             Professur für Energieverfahrenstechnik (EVT)
E-Mail: montadhar.guesmi@tu-dresden.de

Summary:
  Reusable detection / classification utilities for gas-object (bubble) analysis
  in plate heat-exchanger images, including U-Net-based segmentation.
===================================================================================================================
"""

import os
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from shapely.geometry import Polygon, LineString
from shapely.prepared import prep
from skimage.morphology import skeletonize

# =============================================================================
# CONFIGURATION
# =============================================================================
FPS = 2000
DT = 1.0 / FPS
PIXEL_TO_METER = 0.0752 / 1000
MIN_AREA_PX = 6 * 6
MAX_AREA_PX = 896 * 896
ROI_LEFT_PX = 75
ROI_RIGHT_PX = 75
BASE_TOL = 5
REL_TOL = 0.07
IRREGULAR_THRESHOLD = 0.3
IMG_H, IMG_W = 256, 256
_unet_model = None

# ------------------------------------------------------------------------------
# 1. IMAGE I/O
# ------------------------------------------------------------------------------
def load_images(folder: str, background_name: str = "background.bmp", pattern: str = "Bild*.bmp"):
    bg_path = os.path.join(folder, background_name)
    bg = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
    if bg is None:
        raise FileNotFoundError(bg_path)
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    files = [f for f in files if os.path.basename(f).lower() != background_name.lower()]
    if len(files) < 2:
        raise RuntimeError("Need at least two frames for tracking.")
    return bg, files

# ------------------------------------------------------------------------------
# 2. PRE-PROCESSING UTILITIES
# ------------------------------------------------------------------------------
def adaptive_background_subtraction(frame: np.ndarray, background: np.ndarray,
                                    base_tol: float = BASE_TOL, rel_tol: float = REL_TOL) -> np.ndarray:
    diff = cv2.absdiff(frame, background).astype(np.float32)
    dyn = base_tol + rel_tol * background.astype(np.float32)
    diff[diff < dyn] = 0
    return diff.clip(0, 255).astype(np.uint8)

def apply_roi(img: np.ndarray, left: int = ROI_LEFT_PX, right: int = ROI_RIGHT_PX) -> np.ndarray:
    out = img.copy()
    h, w = out.shape
    out[:, :left] = 0
    out[:, w - right:] = 0
    return out

def apply_clahe(img: np.ndarray, clip: float = 1.5, grid: tuple = (12, 12)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(img)

def preprocess_for_unet(rgb_image: np.ndarray, background: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    diff = adaptive_background_subtraction(gray, background)
    diff = apply_roi(diff)
    diff = apply_clahe(diff, clip=2.0, grid=(8, 8))
    diff_color = cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(diff_color, (IMG_W, IMG_H)).astype(np.float32) / 255.0
    return resized

def process_mask(mask: np.ndarray, size: tuple) -> np.ndarray:
    resized = cv2.resize(mask.astype(np.uint8) * 255, size, interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(resized, kernel, iterations=1)
    binary_bool = (dilated > 0).astype(bool)
    skeleton = skeletonize(binary_bool).astype(np.uint8) * 255
    return skeleton

def GaussianFilter_Based_Threshold_Edge_Detection(diff):
    thresh = 65
    gauss = 5
    canny_low = 0
    canny_high = 86
    morph = 5
    morph_iter = 5

    _, binary_mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(diff, (gauss, gauss), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    combined = cv2.bitwise_or(binary_mask, edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)
    return combined

def BilateralFilter_Based_Threshold_Edge_Detection(diff):
    d = 9
    sigmaC = 75
    sigmaS = 75
    thresh = 20
    canny_low = 0
    canny_high = 20
    morph = 5
    morph_iter = 5

    filtered = cv2.bilateralFilter(diff, d=d, sigmaColor=sigmaC, sigmaSpace=sigmaS)
    filtered = cv2.medianBlur(filtered, 11)
    _, binary_mask = cv2.threshold(filtered, thresh, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(filtered, canny_low, canny_high)
    combined = cv2.bitwise_or(binary_mask, edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)
    return closed


# ------------------------------------------------------------------------------
# 3. CONTOUR METRICS & CLASSIFICATION
# ------------------------------------------------------------------------------
def _approx_pts(cnt):
    return [tuple(pt[0]) for pt in cv2.approxPolyDP(cnt, 2.0, True)]

def compute_aspect_ratio(cnt):
    pts = _approx_pts(cnt)
    if len(pts) < 2: return 1.0
    md = max(np.linalg.norm(np.array(p1) - p2) for i, p1 in enumerate(pts) for p2 in pts[i + 1:])
    return md

def compute_circularity(cnt):
    a = cv2.contourArea(cnt)
    p = cv2.arcLength(cnt, True)
    return (4 * np.pi * a / p ** 2) if p else 0

def _downsample(cnt, step=1):
    pts = _approx_pts(cnt)
    return pts if len(pts) < 4 else pts[::step]

def _vis_graph_edges(pts):
    n = len(pts)
    if n < 4: return []
    if pts[0] != pts[-1]: pts = pts + [pts[0]]
    poly_p = prep(Polygon(pts).buffer(0))
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1): continue
            if poly_p.contains(LineString([pts[i], pts[j]])):
                edges.append((i, j))
    return edges

def visibility_complexity(pts):
    e = _vis_graph_edges(pts)
    n = len(pts)
    return len(e) / (n * (n - 1) / 2) if n > 1 else 0.0

def visibility_ratio(pts):
    n = len(pts)
    return len(_vis_graph_edges(pts)) / (n * (n - 3) / 2) if n >= 4 else 1.0

def quantification_metrics(cnt):
    area = cv2.contourArea(cnt)
    if area == 0:
        return None
    ds = _downsample(cnt, step=2 if len(cnt) > 200 else 1)
    comp = visibility_complexity(ds)
    vis = visibility_ratio(ds)
    ar = compute_aspect_ratio(cnt)
    circ = compute_circularity(cnt)
    eqd = math.sqrt(4 * area / np.pi) * PIXEL_TO_METER * 1000
    per = cv2.arcLength(cnt, True)
    return {
        "area": area, "perimeter": per, "complexity_score": comp,
        "visibility_ratio": vis, "aspect_ratio": ar,
        "circularity": circ, "equivalent_diameter": eqd
    }

def classify_shape(cnt, irr_thr: float = IRREGULAR_THRESHOLD):
    m = quantification_metrics(cnt)
    if m is None: return -1, None
    cs, circ, eqd, ar = m["complexity_score"], m["circularity"], m["equivalent_diameter"], m["aspect_ratio"]
    BIG, SMALL, CIRC_T = 8.0, 1.0, 0.7
    if eqd >= BIG:
        return (4 if 0 < cs < irr_thr else 3), m
    if 0 < cs < irr_thr:
        return 5, m
    if eqd <= 4.0:
        if eqd < SMALL:
            return (0 if circ > CIRC_T else -1), m
        return (1 if circ > CIRC_T else 5), m
    return (2 if 1.0 < ar <= 50 else 5), m

# ------------------------------------------------------------------------------
# 4. U-NET-BASED DETECTION (method3)
# ------------------------------------------------------------------------------
def load_unet_model(model_path="final_model_instance_boundary.keras"):
    global _unet_model
    if _unet_model is None:
        _unet_model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "iou_coef": lambda y_true, y_pred: y_pred,
                "bce_dice_loss": lambda y_true, y_pred: y_pred
            }
        )
    return _unet_model

def _close_openings(mask: np.ndarray, cnts):
    """Closes open contours that touch image borders."""
    h, w = mask.shape
    out = mask.copy()
    for c in cnts:
        pts = c[:, 0, :]
        on_border = [tuple(p) for p in pts if p[0] <= 1 or p[0] >= w - 2 or p[1] <= 1 or p[1] >= h - 2]
        if len(on_border) >= 2:
            bp = np.array(on_border)
            d2 = ((bp[:, None, :] - bp[None, :, :]) ** 2).sum(-1)
            i, j = np.unravel_index(np.argmax(d2), d2.shape)
            cv2.line(out, tuple(bp[i]), tuple(bp[j]), 255, 1)
    return out


def detect_method_bubbles(frame_gray: np.ndarray, background: np.ndarray, method: str = "method1",
                          roi_left: int = ROI_LEFT_PX, roi_right: int = ROI_RIGHT_PX) -> np.ndarray:
    diff = adaptive_background_subtraction(frame_gray, background)
    diff = apply_roi(diff, left=roi_left, right=roi_right)
    diff = apply_clahe(diff, clip=2.0, grid=(8, 8))
    if method == "method1":
        return GaussianFilter_Based_Threshold_Edge_Detection(diff)
    elif method == "method2":
        return BilateralFilter_Based_Threshold_Edge_Detection(diff)
    else:
        raise ValueError(f"Unsupported method: {method}")


def detect_and_classify_bubbles(frame: np.ndarray,
                                background: np.ndarray,
                                method: str = "method1",
                                model_path: str = "final_model_instance_boundary.keras"):
    """
    Main bubble detection + classification interface for all methods.

    Parameters:
        frame: np.ndarray – either grayscale (method1/2) or RGB (method3).
        background: np.ndarray – grayscale background image.
        method: str – "method1", "method2", or "method3" (U-Net).
        model_path: str – only used for U-Net model loading.

    Returns:
        dict with detected objects and the binary mask used.
    """
    if method == "method3":
        return detect_unet_bubbles(frame, background, model_path=model_path)

    # For classical method1/method2
    if len(frame.shape) == 3:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        frame_gray = frame.copy()

    mask = detect_method_bubbles(frame_gray, background, method=method)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = _close_openings(mask, cnts)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = {}
    idx = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if MIN_AREA_PX <= area <= MAX_AREA_PX:
            idx += 1
            cls, metrics = classify_shape(c)
            results[idx] = {"contour": c, "id": idx, "classification": cls, "metrics": metrics}
    return results, mask

"""
def detect_unet_bubbles(frame_rgb: np.ndarray, background_gray: np.ndarray, model_path: str = "final_model_instance_boundary.keras"):
    # Ensure input is 3-channel RGB for U-Net
    if len(frame_rgb.shape) == 2 or frame_rgb.shape[2] == 1:
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)
    elif frame_rgb.shape[2] == 3:
        # OpenCV loads color images as BGR by default, convert to RGB
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
    
    model = load_unet_model(model_path)
    input_tensor = preprocess_for_unet(frame_rgb, background_gray)
    pred = model.predict(np.expand_dims(input_tensor, axis=0), verbose=0)
    
    boundary = pred[1][0, ..., 0] if isinstance(pred, list) else pred[0][0, ..., 1]
    mask256 = (boundary > 0.5).astype(np.uint8)

    h, w = frame_rgb.shape[:2]
    skeleton = process_mask(mask256, (w, h))
    num_labels, labels = cv2.connectedComponents(skeleton)

    results = {}
    idx = 0
    for i in range(1, num_labels):
        blob_mask = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = cnts[0]
        area = cv2.contourArea(cnt)
        if MIN_AREA_PX <= area <= MAX_AREA_PX:
            idx += 1
            class_id, metrics = classify_shape(cnt)
            results[idx] = {
                "skeleton": blob_mask,
                "contour": cnt,
                "id": idx,
                "classification": class_id,
                "metrics": metrics
            }

    return results, skeleton
"""

def detect_unet_bubbles(frame_rgb: np.ndarray, background_gray: np.ndarray, model_path: str = "final_model_instance_boundary.keras"):
    # Ensure input is 3-channel RGB for U-Net
    if len(frame_rgb.shape) == 2 or frame_rgb.shape[2] == 1:
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)
    elif frame_rgb.shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)

    model = load_unet_model(model_path)
    input_tensor = preprocess_for_unet(frame_rgb, background_gray)
    pred = model.predict(np.expand_dims(input_tensor, axis=0), verbose=0)

    boundary = pred[1][0, ..., 0] if isinstance(pred, list) else pred[0][0, ..., 1]
    mask256 = (boundary > 0.3).astype(np.uint8)

    h, w = frame_rgb.shape[:2]

    # Use filled binary mask resized to original image size
    filled_mask = cv2.resize(mask256 * 255, (w, h), interpolation=cv2.INTER_NEAREST)

    # Use filled mask for connected component labeling
    num_labels, labels = cv2.connectedComponents(filled_mask)

    results = {}
    idx = 0
    for i in range(1, num_labels):  # skip background (label 0)
        blob_mask = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = cnts[0]
        area = cv2.contourArea(cnt)
        if MIN_AREA_PX <= area <= MAX_AREA_PX:
            idx += 1
            class_id, metrics = classify_shape(cnt)
            results[idx] = {
                "skeleton": blob_mask,  # or optionally: 'filled_mask': blob_mask
                "contour": cnt,
                "id": idx,
                "classification": class_id,
                "metrics": metrics
            }

    return results, filled_mask  # return filled mask for gas fraction, not skeleton


# ------------------------------------------------------------------------------
# 5. COLOR MAPPING FOR VISUALIZATION
# ------------------------------------------------------------------------------
CLUSTER_COLORS = {-1: "red", 0: "darkblue", 1: "green", 2: "yellow", 3: "cyan", 4: "orange", 5: "purple"}
CLUSTER_NAMES = {
    -1: "Artifact", 0: "Round Small Bubble", 1: "Round Coarse Bubble",
     2: "Elongated Bubble", 3: "Big Gas Object (Regular)",
     4: "Big Gas Object (Irregular)", 5: "Irregular Bubble"
}

def map_cluster_to_color(cid):
    return CLUSTER_COLORS.get(cid, "red")

def map_shape_name(cid):
    return CLUSTER_NAMES.get(cid, "Artifact")

def update(frame_idx, background, all_files, ax, model_path):
    for a in ax:
        a.clear()
        a.axis("off")

    fpath = all_files[frame_idx]
    raw_bgr = cv2.imread(fpath, cv2.IMREAD_COLOR)
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)

    # Method 1: Gaussian
    results1, _ = detect_and_classify_bubbles(raw_gray, background, method="method1")
    ax[0].imshow(raw_gray, cmap="gray")
    ax[0].set_title(f"Method 1: Gaussian (Frame {frame_idx})")
    for obj in results1.values():
        contour, cid = obj["contour"], obj["classification"]
        color = map_cluster_to_color(cid)
        contour = contour.squeeze()
        if contour.ndim == 2 and contour.shape[0] > 1:
            ax[0].plot(contour[:, 0], contour[:, 1], color=color, linewidth=1.5)
            ax[0].text(contour[0, 0], contour[0, 1], str(obj["id"]), fontsize=8, color="white")

    # Method 2: Bilateral
    results2, _ = detect_and_classify_bubbles(raw_gray, background, method="method2")
    ax[1].imshow(raw_gray, cmap="gray")
    ax[1].set_title(f"Method 2: Bilateral (Frame {frame_idx})")
    for obj in results2.values():
        contour, cid = obj["contour"], obj["classification"]
        color = map_cluster_to_color(cid)
        contour = contour.squeeze()
        if contour.ndim == 2 and contour.shape[0] > 1:
            ax[1].plot(contour[:, 0], contour[:, 1], color=color, linewidth=1.5)
            ax[1].text(contour[0, 0], contour[0, 1], str(obj["id"]), fontsize=8, color="white")

    # Method 3: U-Net + Skeleton
    results3, skeleton = detect_and_classify_bubbles(raw_rgb, background, method="method3", model_path=model_path)
    ax[2].imshow(raw_gray, cmap="gray")
    ax[2].set_title(f"Method 3: U-Net (Frame {frame_idx})")
    for obj in results3.values():
        contour, cid = obj["contour"], obj["classification"]
        color = map_cluster_to_color(cid)
        contour = contour.squeeze()
        if contour.ndim == 2 and contour.shape[0] > 1:
            ax[2].plot(contour[:, 0], contour[:, 1], color=color, linewidth=1.5)
            ax[2].text(contour[0, 0], contour[0, 1], str(obj["id"]), fontsize=8, color="white")

    # Shared Legend
    legend_patches = [
        plt.Line2D([0], [0], color=map_cluster_to_color(cid), lw=2, label=map_shape_name(cid))
        for cid in range(-1, 6)
    ]
    ax[0].legend(handles=legend_patches, loc="upper right")
    ax[1].legend(handles=legend_patches, loc="upper right")
    ax[2].legend(handles=legend_patches, loc="upper right")

