from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response, HTMLResponse
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os
from collections import defaultdict
import random

app = FastAPI()

# Load the YOLO model
model = YOLO("runs/train/PID_YOLOv8/weights/best.pt")

# Dictionary to store class counters
class_counters = defaultdict(int)

# Generate random color for each class
def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

class_colors = {}

def process_image(image_bytes):
    # Reset class counters for each new image
    class_counters.clear()
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run prediction
    results = model.predict(source=img, conf=0.4, show_labels=False, show_conf=False)
    
    # Get the first result
    result = results[0]
    
    # Create copies for both output types
    img_with_boxes = img.copy()
    detected_objects = []
    
    # Process each detection
    for box in result.boxes:
        # Get class name
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        
        # Increment counter for this class
        class_counters[class_name] += 1
        unique_id = f"{class_name}-{class_counters[class_name]:02d}"
        
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Get or generate color for this class
        if class_name not in class_colors:
            class_colors[class_name] = get_random_color()
        color = class_colors[class_name]
        
        # Draw rectangle
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # Store detection info for HTML
        detected_objects.append({
            'id': unique_id,
            'class': class_name,
            'bbox': [x1, y1, x2, y2],
            'color': f'rgb{color}'
        })
    
    # Convert image to bytes for PNG response
    _, img_encoded = cv2.imencode('.png', img_with_boxes)
    img_bytes = img_encoded.tobytes()
    
    return img_bytes, detected_objects

def generate_html(detected_objects, img_width, img_height):
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .container {{
                position: relative;
                display: inline-block;
            }}
            .bbox {{
                position: absolute;
                border: 2px solid;
                cursor: pointer;
            }}
            .label {{
                position: absolute;
                background: white;
                padding: 2px 5px;
                border: 1px solid black;
                display: none;
            }}
        </style>
        <script>
            function toggleLabel(id) {{
                const label = document.getElementById('label-' + id);
                label.style.display = label.style.display === 'none' ? 'block' : 'none';
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <img src="data:image/png;base64,PLACEHOLDER_IMAGE" width="{img_width}" height="{img_height}">
    """
    
    for obj in detected_objects:
        x1, y1, x2, y2 = obj['bbox']
        width = x2 - x1
        height = y2 - y1
        
        html += f"""
            <div class="bbox" 
                 style="left: {x1}px; top: {y1}px; width: {width}px; height: {height}px; border-color: {obj['color']}"
                 onclick="toggleLabel('{obj['id']}')">
            </div>
            <div class="label" 
                 id="label-{obj['id']}" 
                 style="left: {x1}px; top: {y2}px;">
                {obj['id']}
            </div>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    return html

@app.post("/detect")
async def detect_symbols(file: UploadFile = File(...), format: str = "png"):
    # Read the uploaded image
    image_bytes = await file.read()
    
    # Process the image
    img_bytes, detected_objects = process_image(image_bytes)
    
    if format.lower() == "html":
        # For HTML response, we need to convert the image to base64
        import base64
        img_base64 = base64.b64encode(img_bytes).decode()
        
        # Get image dimensions
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        height, width = img.shape[:2]
        
        # Generate HTML
        html_content = generate_html(detected_objects, width, height)
        html_content = html_content.replace('PLACEHOLDER_IMAGE', img_base64)
        
        return HTMLResponse(content=html_content)
    else:
        # For PNG response
        from fastapi.responses import Response
        return Response(
            content=img_bytes,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=result.png"}
        )