
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse, FileResponse
# import torch
# import torchvision
# from torchvision.ops import nms
# import numpy as np
# from PIL import Image
# import requests
# from fastapi.encoders import jsonable_encoder
# import base64
# from io import BytesIO
# import os
# import matplotlib.pyplot as plt
# import uuid

# app = FastAPI(title="P&ID Symbol Detection API", version="1.3")

# # =====================================================
# # Model Class
# # =====================================================
# class PidDetectionModel:
#     def __init__(self, model_path: str):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Using device: {self.device}")

#         checkpoint = torch.load(model_path, map_location=self.device)
#         from torchvision.models.detection import fasterrcnn_resnet50_fpn

#         self.model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=33)
#         self.model.load_state_dict(checkpoint["state_dict"])
#         self.model.eval()

#         self.class_names = {
#             0: "Pump", 1: "Valve", 2: "Tank", 3: "Heat Exchanger", 4: "Compressor",
#             5: "Motor", 6: "Filter", 7: "Separator", 8: "Reactor", 9: "Distillation Column",
#             10: "Storage Tank", 11: "Control Valve", 12: "Safety Valve", 13: "Check Valve",
#             14: "Gate Valve", 15: "Ball Valve", 16: "Butterfly Valve", 17: "Globe Valve",
#             18: "Relief Valve", 19: "Shut-off Valve", 20: "Flow Meter", 21: "Pressure Gauge",
#             22: "Temperature Sensor", 23: "Level Sensor", 24: "Pressure Transmitter",
#             25: "Flow Transmitter", 26: "Temperature Transmitter", 27: "Level Transmitter",
#             28: "Control Panel", 29: "PLC", 30: "DCS", 31: "SCADA", 32: "Other",
#         }

#     def detect_symbols_optimized(self, image_path, nms_threshold=0.5):
#         if image_path.startswith("http"):
#             response = requests.get(image_path)
#             image = Image.open(BytesIO(response.content))
#         else:
#             image = Image.open(image_path)
#         if image.mode != "RGB":
#             image = image.convert("RGB")

#         img_array = np.array(image)
#         original_height, original_width = img_array.shape[:2]
#         confidence_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
#         target_sizes = [(800, 600), (1024, 768), (1200, 900)]

#         best_detections = None
#         max_detections = 0

#         for confidence_threshold in confidence_thresholds:
#             for target_size in target_sizes:
#                 resized_image = image.resize(target_size, Image.LANCZOS)
#                 image_tensor = torchvision.transforms.functional.to_tensor(resized_image)
#                 image_tensor = image_tensor.unsqueeze(0).to(self.device)

#                 with torch.no_grad():
#                     predictions = self.model(image_tensor)

#                 boxes = predictions[0]["boxes"].cpu().numpy()
#                 scores = predictions[0]["scores"].cpu().numpy()
#                 labels = predictions[0]["labels"].cpu().numpy()

#                 high_conf_mask = scores >= confidence_threshold
#                 filtered_boxes = boxes[high_conf_mask]
#                 filtered_scores = scores[high_conf_mask]
#                 filtered_labels = labels[high_conf_mask]

#                 if len(filtered_boxes) > 0:
#                     keep = nms(torch.tensor(filtered_boxes), torch.tensor(filtered_scores), nms_threshold)
#                     filtered_boxes = filtered_boxes[keep.numpy()]
#                     filtered_scores = filtered_scores[keep.numpy()]
#                     filtered_labels = filtered_labels[keep.numpy()]

#                     scale_x = original_width / target_size[0]
#                     scale_y = original_height / target_size[1]
#                     scaled_boxes = filtered_boxes.copy()
#                     scaled_boxes[:, [0, 2]] *= scale_x
#                     scaled_boxes[:, [1, 3]] *= scale_y

#                     if len(filtered_boxes) > max_detections:
#                         max_detections = len(filtered_boxes)
#                         best_detections = (scaled_boxes, filtered_scores, filtered_labels)

#         return self._process_detections(best_detections) if best_detections else []

#     def _process_detections(self, detections):
#         boxes, scores, labels = detections
#         processed_symbols = []
#         for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
#             x1, y1, x2, y2 = box.astype(int)
#             centroid_x, centroid_y = int((x1 + x2) // 2), int((y1 + y2) // 2)
#             corrected_label = max(0, int(label) - 1)
#             class_name = self.class_names.get(corrected_label, f"Class {corrected_label}")
#             processed_symbols.append({
#                 "id": i,
#                 "class_id": corrected_label,
#                 "class_name": class_name,
#                 "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
#                 "centroid": {"x": centroid_x, "y": centroid_y},
#                 "confidence": float(score),
#                 "tag_number": f"{class_name}_{i+1}"
#             })
#         return processed_symbols

#     def visualize_detections(self, image_path, detections, save_path="result.png"):
#         image = Image.open(image_path).convert("RGB")
#         img_array = np.array(image)
#         fig, ax = plt.subplots(figsize=(15, 10))
#         ax.imshow(img_array)
#         ax.axis("off")

#         colors = plt.cm.tab20(np.linspace(0, 1, len(self.class_names)))

#         for det in detections:
#             bbox = det["bbox"]
#             x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
#             color = colors[det["class_id"] % len(colors)]
#             rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
#                                  fill=False, edgecolor=color, linewidth=2)
#             ax.add_patch(rect)
#             label = f"{det['class_name']} ({det['confidence']:.2f})"
#             ax.text(x1, max(y1 - 10, 0), label, fontsize=8, color="black",
#                     bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.2"))

#         plt.tight_layout()
#         fig.canvas.draw()
#         plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
#         plt.close(fig)
#         return save_path


# # =====================================================
# # Helper: Tile-based detection
# # =====================================================
# def detect_symbols_tiled(detector, image_path, tile_size=1024, overlap=100):
#     image = Image.open(image_path).convert("RGB")
#     img_array = np.array(image)
#     h, w = img_array.shape[:2]
#     detections_all = []

#     for y in range(0, h, tile_size - overlap):
#         for x in range(0, w, tile_size - overlap):
#             tile = img_array[y:y+tile_size, x:x+tile_size]
#             if tile.shape[0] == 0 or tile.shape[1] == 0:
#                 continue
#             tile_img = Image.fromarray(tile)
#             tile_path = "temp_tile.png"
#             tile_img.save(tile_path)
#             detections = detector.detect_symbols_optimized(tile_path)
#             for det in detections:
#                 det["bbox"]["x1"] += x
#                 det["bbox"]["x2"] += x
#                 det["bbox"]["y1"] += y
#                 det["bbox"]["y2"] += y
#                 det["centroid"]["x"] += x
#                 det["centroid"]["y"] += y
#                 detections_all.append(det)
#     return detections_all


# # =====================================================
# # Initialize model
# # =====================================================
# MODEL_DIR = "models"
# MODEL_PATH = os.path.join(MODEL_DIR, "pandid_rcnn.t7")

# if not os.path.exists(MODEL_DIR):
#     os.makedirs(MODEL_DIR, exist_ok=True)
#     print(f"📁 Created model directory: {MODEL_DIR}")

# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(
#         f"❌ Model file not found at: {MODEL_PATH}\n"
#         f"Please place 'pandid_rcnn.t7' inside the 'models/' directory before starting the server."
#     )

# print(f"✅ Found model at: {MODEL_PATH}")
# detector = PidDetectionModel(MODEL_PATH)


# # =====================================================
# # Utility
# # =====================================================
# def clean(obj):
#     if isinstance(obj, (np.integer, np.int32, np.int64)):
#         return int(obj)
#     elif isinstance(obj, (np.floating, np.float32, np.float64)):
#         return float(obj)
#     elif isinstance(obj, dict):
#         return {k: clean(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [clean(v) for v in obj]
#     else:
#         return obj


# # =====================================================
# # API Routes
# # =====================================================
# @app.get("/")
# def root():
#     return {"message": "✅ P&ID Symbol Detection API running!"}


# @app.post("/detect_tiled")
# async def detect_tiled(file: UploadFile = File(...)):
#     image_path = file.filename
#     with open(image_path, "wb") as f:
#         f.write(await file.read())
#     detections = detect_symbols_tiled(detector, image_path)
#     os.remove(image_path)

#     detections_clean = clean(detections)
#     return JSONResponse(content=jsonable_encoder({
#         "count": len(detections_clean),
#         "detections": detections_clean
#     }))


# @app.post("/visualize")
# async def visualize(file: UploadFile = File(...)):
#     image_path = file.filename
#     with open(image_path, "wb") as f:
#         f.write(await file.read())
#     detections = detect_symbols_tiled(detector, image_path)
#     result_path = detector.visualize_detections(image_path, detections, save_path="result.png")
#     os.remove(image_path)
#     return FileResponse(result_path, media_type="image/png", filename="result.png")


# # =====================================================
# # ✅ Unified Endpoint (Detect + Visualize with Download URL)
# # =====================================================
# @app.post("/detect_and_visualize")
# async def detect_and_visualize(file: UploadFile = File(...)):
#     image_path = f"temp_{uuid.uuid4().hex}.png"
#     with open(image_path, "wb") as f:
#         f.write(await file.read())

#     detections = detect_symbols_tiled(detector, image_path)
#     detections_clean = clean(detections)

#     result_filename = f"result_{uuid.uuid4().hex}.png"
#     result_path = detector.visualize_detections(image_path, detections, save_path=result_filename)

#     if not os.path.exists(result_path) or os.path.getsize(result_path) == 0:
#         raise HTTPException(status_code=500, detail="Visualization failed — empty image output.")

#     os.remove(image_path)

#     return JSONResponse(content={
#         "status": "success",
#         "message": "Detection and visualization completed.",
#         "count": len(detections_clean),
#         "detections": detections_clean,
#         "download_url": f"/download/{os.path.basename(result_path)}"
#     })


# # =====================================================
# # 📥 Download Endpoint
# # =====================================================
# @app.get("/download/{filename}")
# async def download_result(filename: str):
#     file_path = os.path.join(os.getcwd(), filename)
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="File not found.")
#     return FileResponse(path=file_path, media_type="image/png", filename=filename)

