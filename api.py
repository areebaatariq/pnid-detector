from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from ultralytics import YOLO
from collections import defaultdict
import cv2
import numpy as np
import random
import uuid
import os

app = FastAPI()

# Load YOLO model
model = YOLO("runs/train/PID_YOLOv8/weights/best.pt")

# Class counters and colors
class_counters = defaultdict(int)
class_colors = {}

def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def process_image(image_bytes):
    class_counters.clear()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = model.predict(source=img, conf=0.4, show_labels=False, show_conf=False)
    result = results[0]
    
    img_with_boxes = img.copy()
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        class_counters[class_name] += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        if class_name not in class_colors:
            class_colors[class_name] = get_random_color()
        color = class_colors[class_name]
        
        label = f"{class_name}-{class_counters[class_name]:02d}"
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return img_with_boxes

@app.post("/detect")
async def detect_symbols(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_with_boxes = process_image(image_bytes)

    # Save the processed image temporarily
    output_path = f"result_{uuid.uuid4().hex}.png"
    cv2.imwrite(output_path, img_with_boxes)

    # Return the image directly (Swagger gives download option)
    return FileResponse(output_path, media_type="image/png", filename="detections.png")
