# file: detect_lines.py

import cv2
import numpy as np
import os

# -----------------------------------------
# Helper: check if a point is inside any symbol bounding box
# -----------------------------------------
def point_in_symbol(x, y, symbols):
    for sym in symbols:
        sx1, sy1, sx2, sy2 = sym["bbox"]
        if sx1 <= x <= sx2 and sy1 <= y <= sy2:
            return True
    return False

# -----------------------------------------
# Helper: create a mask for symbols
# -----------------------------------------
def mask_symbols(image_shape, symbols):
    mask = np.ones(image_shape[:2], dtype=np.uint8) * 255
    for sym in symbols:
        x1, y1, x2, y2 = sym["bbox"]
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)  # black out symbols
    return mask

# -----------------------------------------
# Helper: detect and mask text regions
# -----------------------------------------
def mask_text_regions(gray):
    # Invert & threshold bright text
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological closing to merge letters into blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of potential text areas
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.ones_like(gray, dtype=np.uint8) * 255
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10:  # ignore tiny blobs
            continue
        cv2.rectangle(mask, (x, y), (x+w, y+h), 0, -1)
    return mask

# -----------------------------------------
# MAIN FUNCTION: detect only real pipeline lines
# -----------------------------------------
def detect_pid_lines(image_path, symbols, output_folder="static"):
    """
    Detects true process pipeline lines in a P&ID using symbol + text masking and Hough Transform.

    Args:
        image_path (str): Path to P&ID image
        symbols (list): YOLO-detected symbols (for masking endpoints)
        output_folder (str): folder to save preview

    Returns:
        filtered_lines (list): [[x1,y1,x2,y2], ...]
        output_image_path (str): saved preview
    """

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    original = img.copy()

    # --------------------------
    # STEP 1 — Grayscale + Masking
    # --------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mask symbols
    symbol_mask = mask_symbols(img.shape, symbols)

    # Mask text regions
    text_mask = mask_text_regions(gray)

    # Combine masks
    combined_mask = cv2.bitwise_and(symbol_mask, text_mask)
    gray_masked = cv2.bitwise_and(gray, gray, mask=combined_mask)

    # --------------------------
    # STEP 2 — Optional blur + threshold
    # --------------------------
    blurred = cv2.medianBlur(gray_masked, 7)
    _, binary = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)

    # --------------------------
    # STEP 3 — Skeletonization
    # --------------------------
    try:
        import cv2.ximgproc as ximgproc
        skeleton = ximgproc.thinning(binary)
    except:
        skeleton = binary.copy()

    # --------------------------
    # STEP 4 — Edge detection + Hough
    # --------------------------
    edges = cv2.Canny(skeleton, 50, 150)

    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=50,
        maxLineGap=10
    )

    filtered_lines = []

    # --------------------------
    # STEP 5 — Filter short lines or lines inside symbols
    # --------------------------
    if raw_lines is not None:
        for line in raw_lines:
            x1, y1, x2, y2 = line[0]

            length = np.hypot(x2 - x1, y2 - y1)
            if length < 40:
                continue

            # Ignore lines fully inside symbols
            if point_in_symbol(x1, y1, symbols) and point_in_symbol(x2, y2, symbols):
                continue

            filtered_lines.append([x1, y1, x2, y2])

            # Draw for preview
            cv2.line(original, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # --------------------------
    # STEP 6 — Save preview
    # --------------------------
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f"lines_{os.path.basename(image_path)}")
    cv2.imwrite(out_path, original)

    return filtered_lines, out_path

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    symbols = [
        {"id": "V1", "bbox": [100, 100, 150, 150]},
        {"id": "P1", "bbox": [300, 200, 360, 260]},
    ]
    lines, out = detect_pid_lines("static/sample_pid.png", symbols)
    print("Detected process lines:", len(lines))
    for ln in lines:
        print(ln)
