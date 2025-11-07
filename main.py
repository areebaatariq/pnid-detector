# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import FileResponse
# from ultralytics import YOLO
# from collections import defaultdict
# import cv2
# import numpy as np
# import random
# import uuid
# import os

# app = FastAPI()

# # Load YOLO model
# model = YOLO("best.pt")


# # Class counters and colors
# class_counters = defaultdict(int)
# class_colors = {}

# def get_random_color():
#     return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# def process_image(image_bytes):
#     class_counters.clear()
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
#     results = model.predict(source=img, conf=0.4, show_labels=False, show_conf=False)
#     result = results[0]
    
#     img_with_boxes = img.copy()
#     for box in result.boxes:
#         class_id = int(box.cls[0])
#         class_name = result.names[class_id]
#         class_counters[class_name] += 1
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
        
#         if class_name not in class_colors:
#             class_colors[class_name] = get_random_color()
#         color = class_colors[class_name]
        
#         label = f"{class_name}-{class_counters[class_name]:02d}"
#         cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(img_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
#     return img_with_boxes

# @app.post("/detect")
# async def detect_symbols(file: UploadFile = File(...)):
#     image_bytes = await file.read()
#     img_with_boxes = process_image(image_bytes)

#     # Save the processed image temporarily
#     output_path = f"result_{uuid.uuid4().hex}.png"
#     cv2.imwrite(output_path, img_with_boxes)

#     # Return the image directly (Swagger gives download option)
#     return FileResponse(output_path, media_type="image/png", filename="detections.png")
# //////////////////////////////////////////////////////////////////////////////


# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from ultralytics import YOLO
# import numpy as np
# from collections import defaultdict
# import random
# import uuid
# import cv2
# import os
# import json

# app = FastAPI(title="P&ID Detection")

# # Load your YOLO model
# model = YOLO("best.pt")  # Make sure best.pt is in the same directory or provide full path

# # Create static folder if not exists
# UPLOAD_FOLDER = "static"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Utility
# class_colors = {}
# class_counters = defaultdict(int)

# def get_dark_color():
#     # RGB values from 0 to 150 for darker shades
#     return [random.randint(0, 150) for _ in range(3)]


# def process_image(image_bytes):
#     class_counters.clear()
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
#     results = model.predict(source=img, conf=0.4, show_labels=False, show_conf=False)
#     result = results[0]

#     detections = []

#     for box in result.boxes:
#         class_id = int(box.cls[0])
#         class_name = result.names[class_id]
#         class_counters[class_name] += 1
#         x1, y1, x2, y2 = map(int, box.xyxy[0])

#         if class_name not in class_colors:
#             class_colors[class_name] = get_dark_color() 
#         color = class_colors[class_name]

#         obj_id = f"{class_name}-{class_counters[class_name]:02d}"
#         detections.append({
#             "id": obj_id,
#             "class": class_name,
#             "bbox": [x1, y1, x2, y2],
#             "color": color
#         })

#     # Save image
#     image_filename = f"result_{uuid.uuid4().hex}.png"
#     image_path = os.path.join(UPLOAD_FOLDER, image_filename)
#     cv2.imwrite(image_path, img)

#     return image_filename, detections  # return only filename for HTML path

# # Serve static files
# app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

# # Home page
# @app.get("/", response_class=HTMLResponse)
# async def home():
#     return """
#     <html>
#         <head><title>P&ID Detection</title></head>
#         <body>
#             <h2>Upload P&ID Image</h2>
#             <form action="/detect" enctype="multipart/form-data" method="post">
#                 <input name="file" type="file" required>
#                 <input type="submit" value="Detect">
#             </form>
#         </body>
#     </html>
#     """

# # Detect endpoint
# @app.post("/detect", response_class=HTMLResponse)
# async def detect(file: UploadFile = File(...)):
#     image_bytes = await file.read()
#     image_filename, detections = process_image(image_bytes)
#     html_image_path = f"/static/{image_filename}"
#     detections_json = json.dumps(detections)

#     html_content = f"""
#     <html>
#     <head>
#         <title>P&ID Detection Interactive Demo</title>
#     </head>
#     <body>
#         <h2>P&ID Detection Interactive Demo</h2>
#         <div id="container" style="position:relative; display:inline-block;">
#             <img id="pid-image" src="{html_image_path}" />
#         </div>
#         <script>
#             const container = document.getElementById('container');
#             const img = document.getElementById('pid-image');
#             const detections = {detections_json};

#             img.onload = () => {{
#                 const scaleX = img.width / img.naturalWidth;
#                 const scaleY = img.height / img.naturalHeight;

#                 detections.forEach(d => {{
#                     let box = document.createElement('div');
#                     box.style.position = 'absolute';
#                     box.style.left = (d.bbox[0] * scaleX) + 'px';
#                     box.style.top = (d.bbox[1] * scaleY) + 'px';
#                     box.style.width = ((d.bbox[2]-d.bbox[0]) * scaleX) + 'px';
#                     box.style.height = ((d.bbox[3]-d.bbox[1]) * scaleY) + 'px';
#                     box.style.border = '4px solid rgb(' + d.color[0] + ',' + d.color[1] + ',' + d.color[2] + ')';
#                     box.style.cursor = 'pointer';
#                     box.title = d.id;
#                     box.onmouseover = () => {{
#                         let tooltip = document.createElement('div');
#                         tooltip.innerText = d.id;
#                         tooltip.style.position = 'absolute';
#                         tooltip.style.background = 'rgba(0,0,0,0.7)';
#                         tooltip.style.color = 'white';
#                         tooltip.style.padding = '2px 5px';
#                         tooltip.style.borderRadius = '3px';
#                         tooltip.style.top = (d.bbox[1]*scaleY - 20) + 'px';
#                         tooltip.style.left = (d.bbox[0]*scaleX) + 'px';
#                         tooltip.id = 'tooltip_' + d.id.replace(/\\s+/g, '');
#                         container.appendChild(tooltip);
#                     }};
#                     box.onmouseout = () => {{
#                         let t = document.getElementById('tooltip_' + d.id.replace(/\\s+/g, ''));
#                         if(t) t.remove();
#                     }};
#                     container.appendChild(box);
#                 }});
#             }};
#         </script>
#     </body>
#     </html>
#     """
#     return HTMLResponse(content=html_content)
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware  # ✅ added
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import random
import uuid
import cv2
import os
import json

app = FastAPI(title="P&ID Detection")

# ✅ Allow frontend (CORS fix)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["https://hensis.intellico.works"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your YOLO model
model = YOLO("best.pt")  # Make sure best.pt is in the same directory or provide full path

# Create static folder if not exists
UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Utility
class_colors = {}
class_counters = defaultdict(int)

def get_dark_color():
    # RGB values from 0 to 150 for darker shades
    return [random.randint(0, 150) for _ in range(3)]


def process_image(image_bytes):
    class_counters.clear()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = model.predict(source=img, conf=0.4, show_labels=False, show_conf=False)
    result = results[0]

    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        class_counters[class_name] += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if class_name not in class_colors:
            class_colors[class_name] = get_dark_color() 
        color = class_colors[class_name]

        obj_id = f"{class_name}-{class_counters[class_name]:02d}"
        detections.append({
            "id": obj_id,
            "class": class_name,
            "bbox": [x1, y1, x2, y2],
            "color": color
        })

    # Save image
    image_filename = f"result_{uuid.uuid4().hex}.png"
    image_path = os.path.join(UPLOAD_FOLDER, image_filename)
    cv2.imwrite(image_path, img)

    return image_filename, detections  # return only filename for HTML path


# Serve static files
app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")


# Home page
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head><title>P&ID Detection</title></head>
        <body>
            <h2>Upload P&ID Image</h2>
            <form action="/detect" enctype="multipart/form-data" method="post">
                <input name="file" type="file" required>
                <input type="submit" value="Detect">
            </form>
        </body>
    </html>
    """


# Detect endpoint
@app.post("/detect", response_class=HTMLResponse)
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_filename, detections = process_image(image_bytes)
    html_image_path = f"/static/{image_filename}"
    detections_json = json.dumps(detections)

    html_content = f"""
    <html>
    <head>
        <title>P&ID Detection Interactive Demo</title>
    </head>
    <body>
        <h2>P&ID Detection Interactive Demo</h2>
        <div id="container" style="position:relative; display:inline-block;">
            <img id="pid-image" src="{html_image_path}" />
        </div>
        <script>
            const container = document.getElementById('container');
            const img = document.getElementById('pid-image');
            const detections = {detections_json};

            img.onload = () => {{
                const scaleX = img.width / img.naturalWidth;
                const scaleY = img.height / img.naturalHeight;

                detections.forEach(d => {{
                    let box = document.createElement('div');
                    box.style.position = 'absolute';
                    box.style.left = (d.bbox[0] * scaleX) + 'px';
                    box.style.top = (d.bbox[1] * scaleY) + 'px';
                    box.style.width = ((d.bbox[2]-d.bbox[0]) * scaleX) + 'px';
                    box.style.height = ((d.bbox[3]-d.bbox[1]) * scaleY) + 'px';
                    box.style.border = '4px solid rgb(' + d.color[0] + ',' + d.color[1] + ',' + d.color[2] + ')';
                    box.style.cursor = 'pointer';
                    box.title = d.id;
                    box.onmouseover = () => {{
                        let tooltip = document.createElement('div');
                        tooltip.innerText = d.id;
                        tooltip.style.position = 'absolute';
                        tooltip.style.background = 'rgba(0,0,0,0.7)';
                        tooltip.style.color = 'white';
                        tooltip.style.padding = '2px 5px';
                        tooltip.style.borderRadius = '3px';
                        tooltip.style.top = (d.bbox[1]*scaleY - 20) + 'px';
                        tooltip.style.left = (d.bbox[0]*scaleX) + 'px';
                        tooltip.id = 'tooltip_' + d.id.replace(/\\s+/g, '');
                        container.appendChild(tooltip);
                    }};
                    box.onmouseout = () => {{
                        let t = document.getElementById('tooltip_' + d.id.replace(/\\s+/g, ''));
                        if(t) t.remove();
                    }};
                    container.appendChild(box);
                }});
            }};
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
