# Boundary Detection with U-Net
## Objective
**Boundary detection** using a trained **U-Net** model. It visualizes the predicted boundaries, with each boundary overlaid on a preprocessed grayscale background.
---
## Workflow Overview
### 1. **Preprocessing**
- A static `background.bmp` is used to subtract background noise from each frame.
- **Adaptive subtraction** is applied:
  - Accounts for both base intensity and relative background tolerance.
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) enhances local contrast in the subtracted frame.
### 2. **U-Net Prediction**
- Each frame is resized to 256×256 for inference.
- The U-Net model outputs a boundary probability map.
- Post-processing includes:
  - Thresholding (`> 0.5`)
  - Morphological cleaning
  - Skeletonization using `skeletonize` and `thin`
  - Optional dilation to improve visibility
### 3. **Overlay**
- Predicted skeletons are overlaid in **red** on the contrast-enhanced grayscale frame.
- Frames are stored in a list for animation.

---

## Applications

This animation pipeline is suitable for:

- **Bubble tracking** in multiphase flows
- **Dynamic segmentation** visualization in biological time-lapse imaging
- **Debugging U-Net temporal predictions** in continuous video sequences

---

## Notes

- This implementation assumes that all `.tif` frames and the background image are in the same directory.
- Boundary skeletons are visualized, not filled contours, to emphasize **edge precision**.
- `display(HTML(...))` ensures compatibility in **Jupyter Notebook environments**.

---

## Example Output

> Below this markdown cell, the notebook will render a dynamic animation showing detected bubble boundaries evolving over time.
