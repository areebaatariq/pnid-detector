# # from fastapi import FastAPI, UploadFile, File
# # from fastapi.responses import HTMLResponse
# # from fastapi.staticfiles import StaticFiles
# # from fastapi.middleware.cors import CORSMiddleware  # ✅ added
# # from ultralytics import YOLO
# # import numpy as np
# # from collections import defaultdict
# # import random
# # import uuid
# # import cv2
# # import os
# # import json

# # app = FastAPI(title="P&ID Detection")

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=[
# #         "http://localhost:8000",
# #         "http://localhost:3000",
# #         "http://127.0.0.1:3000",
# #         "http://127.0.0.1:5173",
# #         "https://hensis-dev.intellico.works",
# #         "https://hensis-pnid-dev.intellico.works",
# #         "https://hensis-nextjs.vercel.app"
# #     ],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # Load your YOLO model
# # model = YOLO("best.pt")  # Make sure best.pt is in the same directory or provide full path

# # # Create static folder if not exists
# # UPLOAD_FOLDER = "static"
# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # # Utility
# # class_colors = {}
# # class_counters = defaultdict(int)

# # def get_dark_color():
# #     # RGB values from 0 to 150 for darker shades
# #     return [random.randint(0, 150) for _ in range(3)]


# # def process_image(image_bytes):
# #     class_counters.clear()
# #     nparr = np.frombuffer(image_bytes, np.uint8)
# #     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
# #     results = model.predict(source=img, conf=0.4, show_labels=False, show_conf=False)
# #     result = results[0]

# #     detections = []

# #     for box in result.boxes:
# #         class_id = int(box.cls[0])
# #         class_name = result.names[class_id]
# #         class_counters[class_name] += 1
# #         x1, y1, x2, y2 = map(int, box.xyxy[0])

# #         if class_name not in class_colors:
# #             class_colors[class_name] = get_dark_color() 
# #         color = class_colors[class_name]

# #         obj_id = f"{class_name}-{class_counters[class_name]:02d}"
# #         detections.append({
# #             "id": obj_id,
# #             "class": class_name,
# #             "bbox": [x1, y1, x2, y2],
# #             "color": color
# #         })

# #     # Save image
# #     image_filename = f"result_{uuid.uuid4().hex}.png"
# #     image_path = os.path.join(UPLOAD_FOLDER, image_filename)
# #     cv2.imwrite(image_path, img)

# #     return image_filename, detections  # return only filename for HTML path


# # # Serve static files
# # app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")


# # # Home page
# # @app.get("/", response_class=HTMLResponse)
# # async def home():
# #     return """
# #     <html>
# #         <head><title>P&ID Detection</title></head>
# #         <body>
# #             <h2>Upload P&ID Image</h2>
# #             <form action="/detect" enctype="multipart/form-data" method="post">
# #                 <input name="file" type="file" required>
# #                 <input type="submit" value="Detect">
# #             </form>
# #         </body>
# #     </html>
# #     """

# # # Detect endpoint with dictionary response and error handling
# # @app.post("/detect", response_class=HTMLResponse)
# # async def detect(file: UploadFile = File(...)):
# #     try:
# #         image_bytes = await file.read()
# #         image_filename, detections = process_image(image_bytes)
# #         html_image_path = f"/static/{image_filename}"

# #         # Prepare tag counts
# #         tag_counts = {cls: class_counters[cls] for cls in class_counters}

# #         # Keep the HTML exactly the same
# #         html_content = f"""
# #         <html>
# #         <head>
# #             <title>P&ID Detection Interactive Demo</title>
# #         </head>
# #         <body>
# #             <h2>P&ID Detection Interactive Demo</h2>
# #             <div id="container" style="position:relative; display:inline-block;">
# #                 <img id="pid-image" src="{html_image_path}" />
# #             </div>
# #             <script>
# #                 const container = document.getElementById('container');
# #                 const img = document.getElementById('pid-image');
# #                 const detections = {json.dumps(detections)};

# #                 img.onload = () => {{
# #                     const scaleX = img.width / img.naturalWidth;
# #                     const scaleY = img.height / img.naturalHeight;

# #                     detections.forEach(d => {{
# #                         let box = document.createElement('div');
# #                         box.style.position = 'absolute';
# #                         box.style.left = (d.bbox[0] * scaleX) + 'px';
# #                         box.style.top = (d.bbox[1] * scaleY) + 'px';
# #                         box.style.width = ((d.bbox[2]-d.bbox[0]) * scaleX) + 'px';
# #                         box.style.height = ((d.bbox[3]-d.bbox[1]) * scaleY) + 'px';
# #                         box.style.border = '4px solid rgb(' + d.color[0] + ',' + d.color[1] + ',' + d.color[2] + ')';
# #                         box.style.cursor = 'pointer';
# #                         box.title = d.id;
# #                         box.onmouseover = () => {{
# #                             let tooltip = document.createElement('div');
# #                             tooltip.innerText = d.id;
# #                             tooltip.style.position = 'absolute';
# #                             tooltip.style.background = 'rgba(0,0,0,0.7)';
# #                             tooltip.style.color = 'white';
# #                             tooltip.style.padding = '2px 5px';
# #                             tooltip.style.borderRadius = '3px';
# #                             tooltip.style.top = (d.bbox[1]*scaleY - 20) + 'px';
# #                             tooltip.style.left = (d.bbox[0]*scaleX) + 'px';
# #                             tooltip.id = 'tooltip_' + d.id.replace(/\\s+/g, '');
# #                             container.appendChild(tooltip);
# #                         }};
# #                         box.onmouseout = () => {{
# #                             let t = document.getElementById('tooltip_' + d.id.replace(/\\s+/g, ''));
# #                             if(t) t.remove();
# #                         }};
# #                         container.appendChild(box);
# #                     }});
# #                 }};
# #             </script>
# #         </body>
# #         </html>
# #         """

# #         # Success response
# #         response_data = {
# #             "status": True,
# #             "message": "Detection completed successfully",
# #             "data": [
# #                 {
# #                     "html_content": html_content,
# #                     "tags_count": tag_counts
# #                 }
# #             ],
# #             "errors": []
# #         }

# #     except Exception as e:
# #         # Error handling
# #         response_data = {
# #             "status": False,
# #             "message": "An error occurred during detection.",
# #             "data": [],
# #             "errors": [str(e)]  # Add the error message to the "errors" field
# #         }

# #     return HTMLResponse(content=json.dumps(response_data), media_type="application/json")







# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware
# from ultralytics import YOLO
# import numpy as np
# import cv2
# import os
# import uuid
# import json
# from collections import defaultdict, deque
# from scipy.spatial.distance import euclidean
# import math

# app = FastAPI(title="P&ID Detection")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:8000",
#         "http://localhost:3000",
#         "http://127.0.0.1:3000",
#         "http://127.0.0.1:5173"
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load YOLO model
# model = YOLO("best.pt")

# # Static folder
# UPLOAD_FOLDER = "static"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Utilities
# class_colors = {}
# class_counters = defaultdict(int)

# def get_dark_color():
#     return [np.random.randint(0, 150) for _ in range(3)]

# # ----- Improved Line Processing -----
# class LineSegment:
#     def __init__(self, x1, y1, x2, y2, detection_id=None):
#         self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
#         self.detection_id = detection_id
#         self.length = euclidean((x1, y1), (x2, y2))
#         self.angle = math.atan2(y2 - y1, x2 - x1)
        
#     def get_endpoints(self):
#         return [(self.x1, self.y1), (self.x2, self.y2)]
    
#     def distance_to_point(self, px, py):
#         """Calculate minimum distance from point to line segment"""
#         A = px - self.x1
#         B = py - self.y1
#         C = self.x2 - self.x1
#         D = self.y2 - self.y1
        
#         dot = A * C + B * D
#         len_sq = C * C + D * D
        
#         if len_sq == 0:
#             return euclidean((px, py), (self.x1, self.y1))
        
#         param = dot / len_sq
        
#         if param < 0:
#             xx, yy = self.x1, self.y1
#         elif param > 1:
#             xx, yy = self.x2, self.y2
#         else:
#             xx = self.x1 + param * C
#             yy = self.y1 + param * D
            
#         return euclidean((px, py), (xx, yy))
    
#     def is_collinear_with(self, other_line, angle_threshold=15, distance_threshold=10):
#         """Check if two lines are collinear (similar angle and close)"""
#         angle_diff = abs(self.angle - other_line.angle)
#         angle_diff = min(angle_diff, math.pi - angle_diff)  # Handle wraparound
        
#         if math.degrees(angle_diff) > angle_threshold:
#             return False
            
#         # Check if lines are close to each other
#         dist1 = other_line.distance_to_point(self.x1, self.y1)
#         dist2 = other_line.distance_to_point(self.x2, self.y2)
#         dist3 = self.distance_to_point(other_line.x1, other_line.y1)
#         dist4 = self.distance_to_point(other_line.x2, other_line.y2)
        
#         return min(dist1, dist2, dist3, dist4) <= distance_threshold

# def merge_collinear_lines(line_segments, gap_threshold=15):
#     """Merge collinear line segments that are close together"""
#     merged_lines = []
#     used = [False] * len(line_segments)
    
#     for i, line1 in enumerate(line_segments):
#         if used[i]:
#             continue
            
#         # Start a new merged line group
#         group = [line1]
#         used[i] = True
        
#         # Find all collinear lines
#         for j, line2 in enumerate(line_segments):
#             if used[j] or i == j:
#                 continue
                
#             if line1.is_collinear_with(line2):
#                 # Check if they're close enough to merge
#                 endpoints1 = line1.get_endpoints()
#                 endpoints2 = line2.get_endpoints()
                
#                 min_dist = float('inf')
#                 for p1 in endpoints1:
#                     for p2 in endpoints2:
#                         min_dist = min(min_dist, euclidean(p1, p2))
                
#                 if min_dist <= gap_threshold:
#                     group.append(line2)
#                     used[j] = True
        
#         # Merge the group into a single line
#         if len(group) == 1:
#             merged_lines.append(group[0])
#         else:
#             # Find the extreme points
#             all_points = []
#             for line in group:
#                 all_points.extend(line.get_endpoints())
            
#             # Project all points onto the average line direction
#             avg_angle = np.mean([line.angle for line in group])
#             direction = np.array([math.cos(avg_angle), math.sin(avg_angle)])
            
#             projections = [np.dot(point, direction) for point in all_points]
#             min_idx = np.argmin(projections)
#             max_idx = np.argmax(projections)
            
#             start_point = all_points[min_idx]
#             end_point = all_points[max_idx]
            
#             merged_line = LineSegment(start_point[0], start_point[1], 
#                                     end_point[0], end_point[1])
#             merged_lines.append(merged_line)
    
#     return merged_lines


# # ----- Line-Rectangle Intersection Utility -----
# def line_intersects_rectangle(line, rx1, ry1, rx2, ry2):
#     """
#     Check if a line segment intersects a rectangle.
#     Line: LineSegment object
#     Rectangle: top-left (rx1, ry1), bottom-right (rx2, ry2)
#     """
#     # Line endpoints
#     x1, y1 = line.x1, line.y1
#     x2, y2 = line.x2, line.y2

#     # Rectangle edges as lines
#     edges = [
#         ((rx1, ry1), (rx2, ry1)),  # top
#         ((rx2, ry1), (rx2, ry2)),  # right
#         ((rx2, ry2), (rx1, ry2)),  # bottom
#         ((rx1, ry2), (rx1, ry1))   # left
#     ]

#     def ccw(A, B, C):
#         return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

#     def intersect(A, B, C, D):
#         return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

#     for (p1, p2) in edges:
#         if intersect((x1,y1),(x2,y2),p1,p2):
#             return True
#     # Also check if line is fully inside rectangle
#     if rx1 <= x1 <= rx2 and ry1 <= y1 <= ry2 and rx1 <= x2 <= rx2 and ry1 <= y2 <= ry2:
#         return True

#     return False

# # ----- Updated Graph Construction -----
# def build_connection_graph(symbols, line_segments, connection_threshold=25):
#     """Build a more accurate connection graph based on line intersections with symbol boundaries"""
#     graph = defaultdict(list)
#     symbol_centers = {}
    
#     # Calculate symbol centers and create bounding boxes with padding
#     symbol_boxes = {}
#     for symbol in symbols:
#         x1, y1, x2, y2 = symbol["bbox"]
#         cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#         symbol_centers[symbol["id"]] = (cx, cy)
        
#         # Expand bounding box slightly for connection detection
#         padding = 10
#         symbol_boxes[symbol["id"]] = (x1 - padding, y1 - padding, 
#                                      x2 + padding, y2 + padding)
    
#     # For each line, find which symbols it connects
#     for line in line_segments:
#         connected_symbols = []
        
#         for symbol in symbols:
#             symbol_id = symbol["id"]
#             bx1, by1, bx2, by2 = symbol_boxes[symbol_id]
#             cx, cy = symbol_centers[symbol_id]
            
#             # Check if line passes close to symbol center or intersects symbol box
#             dist_to_center = line.distance_to_point(cx, cy)
            
#             # Also check if line intersects the symbol's bounding box
#             intersects_box = line_intersects_rectangle(line, bx1, by1, bx2, by2)
            
#             if dist_to_center <= connection_threshold or intersects_box:
#                 connected_symbols.append(symbol_id)
        
#         # Connect all pairs of symbols that share this line
#         for i in range(len(connected_symbols)):
#             for j in range(i + 1, len(connected_symbols)):
#                 sym1, sym2 = connected_symbols[i], connected_symbols[j]
#                 if sym2 not in graph[sym1]:
#                     graph[sym1].append(sym2)
#                 if sym1 not in graph[sym2]:
#                     graph[sym2].append(sym1)
    
#     return graph, symbol_centers

# # ----- Path Finding with Line Tracking -----
# # ----- Robust Path Finding -----
# def find_path_with_lines(graph, symbol_centers, line_segments, start_symbols, connection_threshold=25):
#     """
#     Find connected path and return both symbols and the specific lines that connect them.
#     This version is more robust: includes lines that touch at least one symbol in the path
#     and tries to connect all symbols recursively.
#     """
#     # BFS to find all reachable symbols from selected ones
#     visited_symbols = set()
#     queue = deque(start_symbols)
    
#     for symbol in start_symbols:
#         visited_symbols.add(symbol)
    
#     while queue:
#         current = queue.popleft()
#         for neighbor in graph.get(current, []):
#             if neighbor not in visited_symbols:
#                 visited_symbols.add(neighbor)
#                 queue.append(neighbor)
    
#     # Collect lines connecting the visited symbols
#     path_lines = []
#     for line in line_segments:
#         for symbol_id in visited_symbols:
#             cx, cy = symbol_centers[symbol_id]
#             # Include line if it's close to any symbol in path
#             if line.distance_to_point(cx, cy) <= connection_threshold:
#                 path_lines.append(line)
#                 break  # No need to check other symbols for this line
    
#     # Optional: remove duplicates (in case a line is appended multiple times)
#     path_lines = list({(line.x1, line.y1, line.x2, line.y2): line for line in path_lines}.values())
    
#     return list(visited_symbols), path_lines

# # ----- Updated Main Processing Function -----
# def process_image(image_bytes):
#     class_counters.clear()
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     results = model.predict(source=img, conf=0.4, show_labels=False, show_conf=False)
#     result = results[0]

#     detections = []
#     raw_lines = []

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

#         # Collect lines for improved merging
#         if "line" in class_name.lower():
#             raw_lines.append(LineSegment(x1, y1, x2, y2, obj_id))

#     # Apply improved line merging
#     merged_line_segments = merge_collinear_lines(raw_lines)

#     pass

#     # # Draw merged lines on image (with increased width)
#     # for line in merged_line_segments:
#     #     cv2.line(img, (int(line.x1), int(line.y1)), (int(line.x2), int(line.y2)), (0, 0, 255), 4)  # Increased width to 4

#     image_filename = f"result_{uuid.uuid4().hex}.png"
#     image_path = os.path.join(UPLOAD_FOLDER, image_filename)
#     cv2.imwrite(image_path, img)

#     return image_filename, detections, merged_line_segments


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
#     image_filename, detections, merged_line_segments = process_image(image_bytes)
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
#                     container.appendChild(box);
#                 }});
#             }};
#         </script>
#     </body>
#     </html>
#     """
#     return HTMLResponse(content=html_content)

# # ----- Updated Graph endpoint -----
# @app.post("/graph")
# async def graph(file: UploadFile = File(...)):
#     image_bytes = await file.read()
#     _, detections, merged_line_segments = process_image(image_bytes)

#     # Separate symbols from lines
#     symbols = [d for d in detections if "line" not in d["class"].lower()]
    
#     # Build improved connection graph
#     graph, symbol_centers = build_connection_graph(symbols, merged_line_segments)

#     return JSONResponse(content={"graph": graph})

# # ----- Updated Highlight paths endpoint -----
# from fastapi import Form

# @app.post("/highlight-paths")
# async def highlight_paths(
#     file: UploadFile = File(...),
#     selected_symbols: str = Form(...)
# ):
#     # Convert comma-separated string to list
#     selected_symbols = [s.strip() for s in selected_symbols.split(",")]

#     if not selected_symbols:
#         return JSONResponse(content={"error": "No symbols selected"}, status_code=400)

#     # Process image
#     image_bytes = await file.read()
#     image_filename, detections, merged_line_segments = process_image(image_bytes)

#     # Filter symbols
#     symbols = [d for d in detections if "line" not in d["class"].lower()]
#     graph, symbol_centers = build_connection_graph(symbols, merged_line_segments)

#     # Keep only valid selected symbols
#     selected_symbols = [s for s in selected_symbols if s in symbol_centers]
#     if not selected_symbols:
#         return JSONResponse(content={"error": "Selected symbols not found"}, status_code=400)

#     # --- New: Find all lines connecting selected symbols ---
#     threshold = 25
#     path_lines = []
#     for line in merged_line_segments:
#         connected = []
#         for symbol_id in selected_symbols:
#             cx, cy = symbol_centers[symbol_id]
#             if line.distance_to_point(cx, cy) <= threshold:
#                 connected.append(symbol_id)
#         if len(connected) >= 2:  # line touches 2+ selected symbols
#             path_lines.append(line)

#     # Highlight lines on image
#     img = cv2.imread(os.path.join(UPLOAD_FOLDER, image_filename))
#     for line in path_lines:
#         cv2.line(img, (int(line.x1), int(line.y1)), (int(line.x2), int(line.y2)), (0,0,255), 4)

#     highlighted_filename = f"highlighted_{uuid.uuid4().hex}.png"
#     highlighted_path = os.path.join(UPLOAD_FOLDER, highlighted_filename)
#     cv2.imwrite(highlighted_path, img)

#     # Return connected symbols, lines, and highlighted image
#     connected_lines = [[int(line.x1), int(line.y1), int(line.x2), int(line.y2)] for line in path_lines]
#     return JSONResponse(content={
#         "connected_symbols": selected_symbols,
#         "connected_lines": connected_lines,
#         "highlighted_image": f"/static/{highlighted_filename}"
#     })

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
import os
import uuid
import json
import logging
from collections import defaultdict, deque
from scipy.spatial.distance import euclidean
import math
from typing import List, Dict, Tuple, Optional
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="P&ID Symbol Detection and Path Highlighting System",
    description="Advanced P&ID symbol detection with YOLOv8 and intelligent path highlighting",
    version="1.0.0"
)

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:5000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Configuration
class Config:
    MODEL_PATH = "best.pt"
    UPLOAD_FOLDER = "static"
    MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    CONNECTION_THRESHOLD = 30
    LINE_THICKNESS = 4
    CONFIDENCE_THRESHOLD = 0.4
    ANGLE_THRESHOLD = 15
    DISTANCE_THRESHOLD = 10
    GAP_THRESHOLD = 20

# Initialize components
try:
    model = YOLO(Config.MODEL_PATH)
    logger.info(f"YOLOv8 model loaded successfully from {Config.MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load YOLOv8 model: {e}")
    raise RuntimeError(f"Model loading failed: {e}")

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Global storage for uploaded files (in production, use Redis or database)
uploaded_files_cache = {}
class_counters = defaultdict(int)
class_colors = {}
def get_unique_dark_color() -> List[int]:
    """Generate a unique dark color for symbol classes"""
    return [np.random.randint(0, 150) for _ in range(3)]

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in Config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not allowed. Allowed types: {', '.join(Config.ALLOWED_EXTENSIONS)}"
        )

class LineSegment:
    """Enhanced line segment class with advanced geometric operations"""
    
    def __init__(self, x1: float, y1: float, x2: float, y2: float, detection_id: Optional[str] = None):
        self.x1, self.y1, self.x2, self.y2 = float(x1), float(y1), float(x2), float(y2)
        self.detection_id = detection_id
        self.length = euclidean((x1, y1), (x2, y2))
        self.angle = math.atan2(y2 - y1, x2 - x1)
        
    def get_endpoints(self) -> List[Tuple[float, float]]:
        """Get line endpoints"""
        return [(self.x1, self.y1), (self.x2, self.y2)]
    
    def get_midpoint(self) -> Tuple[float, float]:
        """Get line midpoint"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def distance_to_point(self, px: float, py: float) -> float:
        """Calculate minimum distance from point to line segment"""
        A = px - self.x1
        B = py - self.y1
        C = self.x2 - self.x1
        D = self.y2 - self.y1
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        
        if len_sq == 0:
            return euclidean((px, py), (self.x1, self.y1))
        
        param = dot / len_sq
        
        if param < 0:
            xx, yy = self.x1, self.y1
        elif param > 1:
            xx, yy = self.x2, self.y2
        else:
            xx = self.x1 + param * C
            yy = self.y1 + param * D
            
        return euclidean((px, py), (xx, yy))
    
    def is_collinear_with(self, other_line: 'LineSegment', 
                         angle_threshold: float = Config.ANGLE_THRESHOLD, 
                         distance_threshold: float = Config.DISTANCE_THRESHOLD) -> bool:
        """Check if two lines are collinear (similar angle and close)"""
        try:
            angle_diff = abs(self.angle - other_line.angle)
            angle_diff = min(angle_diff, math.pi - angle_diff)  # Handle wraparound
            
            if math.degrees(angle_diff) > angle_threshold:
                return False
                
            # Check if lines are close to each other
            distances = [
                other_line.distance_to_point(self.x1, self.y1),
                other_line.distance_to_point(self.x2, self.y2),
                self.distance_to_point(other_line.x1, other_line.y1),
                self.distance_to_point(other_line.x2, other_line.y2)
            ]
            
            return min(distances) <= distance_threshold
        except Exception as e:
            logger.warning(f"Error in collinearity check: {e}")
            return False

def merge_collinear_lines(line_segments: List[LineSegment], 
                         gap_threshold: float = Config.GAP_THRESHOLD) -> List[LineSegment]:
    """Merge collinear line segments that are close together"""
    if not line_segments:
        return []
    
    try:
        merged_lines = []
        used = [False] * len(line_segments)
        
        for i, line1 in enumerate(line_segments):
            if used[i]:
                continue
                
            # Start a new merged line group
            group = [line1]
            used[i] = True
            
            # Find all collinear lines
            for j, line2 in enumerate(line_segments):
                if used[j] or i == j:
                    continue
                    
                if line1.is_collinear_with(line2):
                    # Check if they're close enough to merge
                    endpoints1 = line1.get_endpoints()
                    endpoints2 = line2.get_endpoints()
                    
                    min_dist = min(
                        euclidean(p1, p2) 
                        for p1 in endpoints1 
                        for p2 in endpoints2
                    )
                    
                    if min_dist <= gap_threshold:
                        group.append(line2)
                        used[j] = True
            
            # Merge the group into a single line
            if len(group) == 1:
                merged_lines.append(group[0])
            else:
                # Find the extreme points
                all_points = []
                for line in group:
                    all_points.extend(line.get_endpoints())
                
                # Project all points onto the average line direction
                avg_angle = np.mean([line.angle for line in group])
                direction = np.array([math.cos(avg_angle), math.sin(avg_angle)])
                
                projections = [np.dot(point, direction) for point in all_points]
                min_idx = np.argmin(projections)
                max_idx = np.argmax(projections)
                
                start_point = all_points[min_idx]
                end_point = all_points[max_idx]
                
                merged_line = LineSegment(start_point[0], start_point[1],
                                        end_point[0], end_point[1])
                merged_lines.append(merged_line)
        
        return merged_lines
    except Exception as e:
        logger.error(f"Error in line merging: {e}")
        return line_segments

def line_intersects_rectangle(line: LineSegment, 
                            rx1: float, ry1: float, 
                            rx2: float, ry2: float) -> bool:
    """
    Check if a line segment intersects a rectangle using robust algorithm.
    Line: LineSegment object
    Rectangle: coordinates (rx1, ry1) to (rx2, ry2)
    """
    try:
        x1, y1, x2, y2 = line.x1, line.y1, line.x2, line.y2
        
        # Ensure rectangle coordinates are ordered correctly
        rx1, rx2 = min(rx1, rx2), max(rx1, rx2)
        ry1, ry2 = min(ry1, ry2), max(ry1, ry2)
        
        # Rectangle edges as line segments
        edges = [
            ((rx1, ry1), (rx2, ry1)),  # top
            ((rx2, ry1), (rx2, ry2)),  # right
            ((rx2, ry2), (rx1, ry2)),  # bottom
            ((rx1, ry2), (rx1, ry1))   # left
        ]
        
        def ccw(A, B, C):
            """Counter-clockwise test"""
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        def intersect(A, B, C, D):
            """Check if line segments AB and CD intersect"""
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
        
        # Check intersection with each edge
        for (p1, p2) in edges:
            if intersect((x1, y1), (x2, y2), p1, p2):
                return True
        
        # Check if line is fully inside rectangle
        if (rx1 <= x1 <= rx2 and ry1 <= y1 <= ry2 and 
            rx1 <= x2 <= rx2 and ry1 <= y2 <= ry2):
            return True
            
        return False
    except Exception as e:
        logger.warning(f"Error in line-rectangle intersection: {e}")
        return False

def build_connection_graph(symbols: List[Dict], 
                         line_segments: List[LineSegment]) -> Tuple[Dict, Dict]:
    """
    Build a connection graph where nodes are symbols and edges are lines.
    Uses both distance-based and intersection-based connection detection.
    """
    try:
        graph = defaultdict(list)
        symbol_centers = {}
        symbol_boxes = {}
        
        # Calculate symbol centers and create expanded bounding boxes
        for symbol in symbols:
            x1, y1, x2, y2 = symbol["bbox"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            symbol_centers[symbol["id"]] = (cx, cy)
            
            # Expand bounding box for connection detection
            padding = 15
            symbol_boxes[symbol["id"]] = (
                x1 - padding, y1 - padding,
                x2 + padding, y2 + padding
            )
        
        # For each line, find which symbols it connects
        for line in line_segments:
            connected_symbols = []
            
            for symbol in symbols:
                symbol_id = symbol["id"]
                bx1, by1, bx2, by2 = symbol_boxes[symbol_id]
                cx, cy = symbol_centers[symbol_id]
                
                # Check distance to symbol center
                dist_to_center = line.distance_to_point(cx, cy)
                
                # Check if line intersects the expanded symbol bounding box
                intersects_box = line_intersects_rectangle(line, bx1, by1, bx2, by2)
                
                if dist_to_center <= Config.CONNECTION_THRESHOLD or intersects_box:
                    connected_symbols.append(symbol_id)
            
            # Connect all pairs of symbols that share this line
            for i in range(len(connected_symbols)):
                for j in range(i + 1, len(connected_symbols)):
                    sym1, sym2 = connected_symbols[i], connected_symbols[j]
                    if sym2 not in graph[sym1]:
                        graph[sym1].append(sym2)
                    if sym1 not in graph[sym2]:
                        graph[sym2].append(sym1)
        
        return dict(graph), symbol_centers
    except Exception as e:
        logger.error(f"Error building connection graph: {e}")
        return {}, {}

def find_all_connected_symbols_and_lines(graph: Dict, symbol_centers: Dict, 
                                       line_segments: List[LineSegment], 
                                       start_symbols: List[str]) -> Tuple[List[str], List[LineSegment]]:
    """
    Find all symbols connected to the selected symbols and all lines connecting them.
    Uses BFS to traverse the connection graph.
    """
    try:
        # BFS to find all reachable symbols from selected ones
        visited_symbols = set()
        queue = deque(start_symbols)
        
        for symbol in start_symbols:
            if symbol in symbol_centers:  # Validate symbol exists
                visited_symbols.add(symbol)
        
        while queue:
            current = queue.popleft()
            for neighbor in graph.get(current, []):
                if neighbor not in visited_symbols:
                    visited_symbols.add(neighbor)
                    queue.append(neighbor)
        
        # Collect lines that connect any symbols in the connected component
        connecting_lines = []
        for line in line_segments:
            line_connects = []
            for symbol_id in visited_symbols:
                if symbol_id in symbol_centers:
                    cx, cy = symbol_centers[symbol_id]
                    # Include line if it's close to any symbol in the connected component
                    if line.distance_to_point(cx, cy) <= Config.CONNECTION_THRESHOLD:
                        line_connects.append(symbol_id)
            
            # Include line if it connects at least 2 symbols in our component
            if len(line_connects) >= 2:
                connecting_lines.append(line)
        
        return list(visited_symbols), connecting_lines
    except Exception as e:
        logger.error(f"Error finding connected symbols: {e}")
        return start_symbols, []

def process_image(image_bytes: bytes) -> Tuple[str, List[Dict], List[LineSegment]]:
    """
    Process uploaded image with YOLOv8 detection and line processing.
    Returns processed image filename, detections, and merged line segments.
    """
    try:
        class_counters.clear()
        
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
        
        logger.info(f"Processing image of size: {img.shape}")
        
        # Run YOLOv8 detection
        results = model.predict(
            source=img, 
            conf=Config.CONFIDENCE_THRESHOLD, 
            show_labels=False, 
            show_conf=False,
            verbose=False
        )
        result = results[0]
        
        detections = []
        raw_lines = []
        
        # Process detection results
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            class_counters[class_name] += 1
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Assign unique color for each symbol class
            if class_name not in class_colors:
                class_colors[class_name] = get_unique_dark_color()
            color = class_colors[class_name]
            
            obj_id = f"{class_name}-{class_counters[class_name]:02d}"
            
            detection = {
                "id": obj_id,
                "class": class_name,
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "color": color
            }
            detections.append(detection)
            
            # Collect lines for processing
            if "line" in class_name.lower():
                raw_lines.append(LineSegment(x1, y1, x2, y2, obj_id))
        
        # Apply intelligent line merging
        merged_line_segments = merge_collinear_lines(raw_lines)
        
        # Draw detection results on image (bounding boxes only, no labels)
        for detection in detections:
            if "line" not in detection["class"].lower():
                x1, y1, x2, y2 = detection["bbox"]
                color = tuple(map(int, detection["color"]))
                
                # Draw bounding box only (no labels by default)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Save processed image
        image_filename = f"result_{uuid.uuid4().hex}.png"
        image_path = os.path.join(Config.UPLOAD_FOLDER, image_filename)
        cv2.imwrite(image_path, img)
        
        logger.info(f"Processed image: {len(detections)} detections, {len(merged_line_segments)} merged lines")
        return image_filename, detections, merged_line_segments
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

# API Endpoints
app.mount("/static", StaticFiles(directory=Config.UPLOAD_FOLDER), name="static")
app.mount("/samples", StaticFiles(directory="samples"), name="samples")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Enhanced home page with comprehensive P&ID detection interface"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>P&ID Symbol Detection & Path Highlighting</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f7fa; }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .header { text-align: center; margin-bottom: 40px; }
            .header h1 { color: #2c3e50; margin-bottom: 10px; font-size: 2.5rem; }
            .header p { color: #7f8c8d; font-size: 1.2rem; }
            .upload-section { background: white; border-radius: 15px; padding: 40px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); margin-bottom: 30px; }
            .upload-area { border: 3px dashed #3498db; border-radius: 15px; padding: 60px 20px; text-align: center; transition: all 0.3s ease; cursor: pointer; }
            .upload-area:hover { border-color: #2980b9; background: #f8f9fa; }
            .upload-icon { font-size: 4rem; color: #3498db; margin-bottom: 20px; }
            .upload-text { color: #2c3e50; font-size: 1.3rem; margin-bottom: 10px; }
            .upload-subtext { color: #7f8c8d; font-size: 0.95rem; }
            .file-input { display: none; }
            .submit-btn { background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 15px 40px; border: none; border-radius: 25px; font-size: 1.1rem; cursor: pointer; margin-top: 25px; transition: all 0.3s ease; }
            .submit-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4); }
            .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 25px; margin-top: 40px; }
            .feature-card { background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.1); text-align: center; }
            .feature-icon { font-size: 2.5rem; margin-bottom: 15px; }
            .feature-title { color: #2c3e50; font-size: 1.2rem; margin-bottom: 10px; font-weight: 600; }
            .feature-desc { color: #7f8c8d; line-height: 1.6; }
            .tech-specs { background: #ecf0f1; padding: 20px; border-radius: 10px; margin-top: 30px; }
            .tech-specs h3 { color: #2c3e50; margin-bottom: 15px; }
            .tech-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
            .tech-item { color: #34495e; padding: 5px 0; }
            .sample-section { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 30px; margin-bottom: 30px; color: white; }
            .sample-section h3 { margin-bottom: 20px; font-size: 1.5rem; }
            .sample-content { display: flex; align-items: center; gap: 30px; flex-wrap: wrap; }
            .sample-preview { max-width: 300px; border-radius: 10px; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
            .sample-info { flex: 1; min-width: 250px; }
            .sample-info p { margin-bottom: 15px; opacity: 0.95; line-height: 1.6; }
            .sample-buttons { display: flex; gap: 15px; flex-wrap: wrap; }
            .btn-sample { padding: 12px 25px; border-radius: 25px; font-size: 1rem; cursor: pointer; transition: all 0.3s ease; text-decoration: none; display: inline-block; font-weight: 600; }
            .btn-download { background: white; color: #667eea; border: none; }
            .btn-download:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(255,255,255,0.4); }
            .btn-try { background: rgba(255,255,255,0.2); color: white; border: 2px solid white; }
            .btn-try:hover { background: white; color: #667eea; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔧 P&ID Detection System</h1>
                <p>Advanced symbol detection and intelligent path highlighting for Piping & Instrumentation Diagrams</p>
            </div>
            
            <div class="sample-section">
                <h3>🧪 Try with Sample Image</h3>
                <div class="sample-content">
                    <img src="/samples/sample_pnid.png" alt="Sample P&ID Diagram" class="sample-preview">
                    <div class="sample-info">
                        <p>Don't have a P&ID image? No problem! Download our sample diagram to test the detection system. This sample includes various symbols like valves, pumps, tanks, and connecting lines.</p>
                        <div class="sample-buttons">
                            <a href="/samples/sample_pnid.png" download="sample_pnid.png" class="btn-sample btn-download">⬇️ Download Sample</a>
                            <button onclick="tryWithSample()" class="btn-sample btn-try">🚀 Try with Sample</button>
                        </div>
                    </div>
                </div>
            </div>

            <div class="upload-section">
                <form id="upload-form" action="/detect" enctype="multipart/form-data" method="post">
                    <div class="upload-area" onclick="document.getElementById('file-input').click()">
                        <div class="upload-icon">📁</div>
                        <div class="upload-text">Click to upload your P&ID image</div>
                        <div class="upload-subtext">Supports JPEG, PNG, BMP, TIFF/TIF (max 20MB)</div>
                        <input id="file-input" name="file" type="file" class="file-input" accept=".jpg,.jpeg,.png,.bmp,.tiff,.tif" required>
                    </div>
                    <div style="text-align: center;">
                        <button type="submit" class="submit-btn">🚀 Start Detection</button>
                    </div>
                </form>
            </div>

            <div class="features">
                <div class="feature-card">
                    <div class="feature-icon">🎯</div>
                    <div class="feature-title">YOLOv8 Detection</div>
                    <div class="feature-desc">State-of-the-art deep learning model for accurate P&ID symbol recognition with customizable confidence thresholds</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔗</div>
                    <div class="feature-title">Smart Line Processing</div>
                    <div class="feature-desc">Advanced algorithms merge collinear line segments and detect complex symbol connections automatically</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🌐</div>
                    <div class="feature-title">Interactive Selection</div>
                    <div class="feature-desc">Click and select multiple symbols to highlight all connecting paths and visualize system relationships</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <div class="feature-title">Graph Analysis</div>
                    <div class="feature-desc">Build connection graphs to analyze symbol relationships and identify complete flow paths</div>
                </div>
            </div>

            <div class="tech-specs">
                <h3>📋 Technical Specifications</h3>
                <div class="tech-list">
                    <div class="tech-item">✓ YOLOv8 Object Detection</div>
                    <div class="tech-item">✓ FastAPI Backend</div>
                    <div class="tech-item">✓ OpenCV Image Processing</div>
                    <div class="tech-item">✓ Geometric Line Analysis</div>
                    <div class="tech-item">✓ Interactive Web Interface</div>
                    <div class="tech-item">✓ RESTful API Endpoints</div>
                </div>
            </div>
        </div>

        <script>
            // File input validation and preview
            document.getElementById('file-input').addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    const uploadText = document.querySelector('.upload-text');
                    uploadText.textContent = `Selected: ${file.name}`;
                    uploadText.style.color = '#27ae60';
                }
            });

            // Try with sample image
            async function tryWithSample() {
                try {
                    // Show loading state
                    const btn = document.querySelector('.btn-try');
                    const originalText = btn.innerHTML;
                    btn.innerHTML = '⏳ Loading...';
                    btn.disabled = true;

                    // Fetch the sample image
                    const response = await fetch('/samples/sample_pnid.png');
                    const blob = await response.blob();
                    
                    // Create a File object from the blob
                    const file = new File([blob], 'sample_pnid.png', { type: 'image/png' });
                    
                    // Create FormData and submit
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    // Submit to detect endpoint
                    const detectResponse = await fetch('/detect', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (detectResponse.ok) {
                        // Replace the page content with the detection result
                        const html = await detectResponse.text();
                        document.open();
                        document.write(html);
                        document.close();
                    } else {
                        throw new Error('Detection failed');
                    }
                } catch (error) {
                    console.error('Error:', error);
                    alert('Failed to process sample image. Please try again.');
                    const btn = document.querySelector('.btn-try');
                    btn.innerHTML = '🚀 Try with Sample';
                    btn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/detect", response_class=HTMLResponse)
async def detect(file: UploadFile = File(...)):
    """
    Detect symbols in P&ID image and return interactive interface.
    Provides comprehensive detection results with symbol selection capabilities.
    """
    try:
        validate_file(file)
        
        image_bytes = await file.read()
        if len(image_bytes) > Config.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File size exceeds 20MB limit")
        
        # Generate unique session ID and cache the file data
        session_id = str(uuid.uuid4())
        uploaded_files_cache[session_id] = {
            'filename': file.filename,
            'content_type': file.content_type,
            'data': image_bytes
        }
        
        image_filename, detections, merged_line_segments = process_image(image_bytes)
        html_image_path = f"/static/{image_filename}"
        
        # Separate symbols from lines for display
        symbols = [d for d in detections if "line" not in d["class"].lower()]
        lines = [d for d in detections if "line" in d["class"].lower()]
        
        detections_json = json.dumps(symbols)
        
        # Enhanced interactive HTML with advanced features
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>P&ID Detection Results</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f7fa; padding: 20px; }}
                .container {{ max-width: 1400px; margin: 0 auto; }}
                .header {{ background: white; padding: 25px; border-radius: 15px; margin-bottom: 25px; box-shadow: 0 4px 16px rgba(0,0,0,0.1); }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 25px; }}
                .stat-card {{ background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 20px; border-radius: 12px; text-align: center; }}
                .stat-number {{ font-size: 2rem; font-weight: bold; margin-bottom: 5px; }}
                .stat-label {{ font-size: 0.9rem; opacity: 0.9; }}
                .controls {{ background: white; padding: 25px; border-radius: 15px; margin-bottom: 25px; box-shadow: 0 4px 16px rgba(0,0,0,0.1); }}
                .button-group {{ display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 20px; }}
                .btn {{ padding: 12px 24px; border: none; border-radius: 25px; cursor: pointer; font-size: 1rem; transition: all 0.3s ease; text-decoration: none; display: inline-block; }}
                .btn-primary {{ background: linear-gradient(135deg, #27ae60, #229954); color: white; }}
                .btn-secondary {{ background: linear-gradient(135deg, #e74c3c, #c0392b); color: white; }}
                .btn-info {{ background: linear-gradient(135deg, #f39c12, #e67e22); color: white; }}
                .btn:hover {{ transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }}
                .btn:disabled {{ background: #bdc3c7; cursor: not-allowed; transform: none; }}
                .selected-display {{ background: #fff3cd; border: 2px solid #ffc107; padding: 15px; border-radius: 10px; margin: 15px 0; }}
                .image-container {{ background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 16px rgba(0,0,0,0.1); margin-bottom: 25px; }}
                #container {{ position: relative; display: inline-block; border: 3px solid #ddd; border-radius: 10px; overflow: hidden; }}
                #pid-image {{ max-width: 100%; height: auto; display: block; }}
                .symbol-box {{ position: absolute; cursor: pointer; transition: all 0.3s ease; }}
                .symbol-box:hover {{ opacity: 0.8; transform: scale(1.05); }}
                .symbol-tooltip {{ 
                    position: absolute; 
                    background: rgba(0, 0, 0, 0.9); 
                    color: white; 
                    padding: 8px 12px; 
                    border-radius: 6px; 
                    font-size: 0.9rem; 
                    font-weight: bold;
                    pointer-events: none; 
                    z-index: 1000; 
                    opacity: 0; 
                    transition: opacity 0.3s ease;
                    white-space: nowrap;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                }}
                .symbol-tooltip.visible {{ opacity: 1; }}
                .symbol-tooltip::after {{ 
                    content: ''; 
                    position: absolute; 
                    top: 100%; 
                    left: 50%; 
                    margin-left: -5px; 
                    border-width: 5px; 
                    border-style: solid; 
                    border-color: rgba(0, 0, 0, 0.9) transparent transparent transparent; 
                }}
                .symbol-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; margin-top: 25px; }}
                .symbol-card {{ background: white; padding: 20px; border-radius: 12px; border: 2px solid #ecf0f1; cursor: pointer; transition: all 0.3s ease; }}
                .symbol-card:hover {{ border-color: #3498db; box-shadow: 0 4px 16px rgba(52, 152, 219, 0.2); }}
                .symbol-card.selected {{ border-color: #f1c40f; background: #fffbf0; }}
                .symbol-title {{ font-weight: bold; color: #2c3e50; margin-bottom: 8px; }}
                .symbol-info {{ color: #7f8c8d; font-size: 0.9rem; }}
                .loading {{ text-align: center; padding: 20px; }}
                .spinner {{ border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 15px; }}
                @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 P&ID Detection Results</h1>
                    <p>Interactive symbol detection and path highlighting interface</p>
                </div>
                
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-number">{len(symbols)}</div>
                        <div class="stat-label">Symbols Detected</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{len(lines)}</div>
                        <div class="stat-label">Lines Detected</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{len(merged_line_segments)}</div>
                        <div class="stat-label">Merged Lines</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="selected-count">0</div>
                        <div class="stat-label">Selected Symbols</div>
                    </div>
                </div>

                <div class="controls">
                    <div class="button-group">
                        <button class="btn btn-secondary" onclick="clearSelection()">🗑️ Clear Selection</button>
                        <button class="btn btn-primary" onclick="highlightPaths()" id="highlight-btn" disabled>🔗 Highlight Paths</button>
                        <button class="btn btn-info" onclick="selectAll()">✅ Select All</button>
                        <a href="/" class="btn btn-info">🏠 New Upload</a>
                    </div>
                    
                    <div class="selected-display" id="selected-display" style="display: none;">
                        <strong>🎯 Selected Symbols:</strong> <span id="selected-list"></span>
                    </div>
                </div>

                <div class="image-container">
                    <div id="container">
                        <img id="pid-image" src="{html_image_path}" alt="P&ID Detection Results" />
                    </div>
                </div>

                <div>
                    <h3>📋 Detected Symbols (Click to Select)</h3>
                    <div class="symbol-grid" id="symbol-grid">
                        <!-- Symbols will be populated by JavaScript -->
                    </div>
                </div>
            </div>

            <script>
                const container = document.getElementById('container');
                const img = document.getElementById('pid-image');
                const detections = {detections_json};
                const sessionId = '{session_id}';
                let selectedSymbols = [];
                
                function updateUI() {{
                    const selectedCount = document.getElementById('selected-count');
                    const selectedDisplay = document.getElementById('selected-display');
                    const selectedList = document.getElementById('selected-list');
                    const highlightBtn = document.getElementById('highlight-btn');
                    
                    selectedCount.textContent = selectedSymbols.length;
                    
                    if (selectedSymbols.length === 0) {{
                        selectedDisplay.style.display = 'none';
                        highlightBtn.disabled = true;
                    }} else {{
                        selectedDisplay.style.display = 'block';
                        selectedList.textContent = selectedSymbols.join(', ');
                        highlightBtn.disabled = false;  // Allow single symbol selection
                    }}
                    
                    // Update symbol cards
                    document.querySelectorAll('.symbol-card').forEach(card => {{
                        const symbolId = card.dataset.symbolId;
                        if (selectedSymbols.includes(symbolId)) {{
                            card.classList.add('selected');
                        }} else {{
                            card.classList.remove('selected');
                        }}
                    }});
                    
                    // Update bounding boxes
                    document.querySelectorAll('.symbol-box').forEach(box => {{
                        const symbolId = box.dataset.symbolId;
                        if (selectedSymbols.includes(symbolId)) {{
                            box.style.backgroundColor = 'rgba(241, 196, 15, 0.4)';
                            box.style.borderWidth = '4px';
                        }} else {{
                            box.style.backgroundColor = 'transparent';
                            box.style.borderWidth = '3px';
                        }}
                    }});
                }}
                
                function toggleSymbolSelection(symbolId) {{
                    const index = selectedSymbols.indexOf(symbolId);
                    if (index > -1) {{
                        selectedSymbols.splice(index, 1);
                    }} else {{
                        selectedSymbols.push(symbolId);
                    }}
                    updateUI();
                }}
                
                function clearSelection() {{
                    selectedSymbols = [];
                    updateUI();
                }}
                
                function selectAll() {{
                    selectedSymbols = detections.map(d => d.id);
                    updateUI();
                }}
                
                async function highlightPaths() {{
                    if (selectedSymbols.length === 0) {{
                        alert('Please select at least 1 symbol to highlight paths');
                        return;
                    }}
                    
                    const highlightBtn = document.getElementById('highlight-btn');
                    const originalText = highlightBtn.innerHTML;
                    highlightBtn.innerHTML = '<div class="spinner"></div> Processing...';
                    highlightBtn.disabled = true;
                    
                    try {{
                        const formData = new FormData();
                        formData.append('session_id', sessionId);
                        formData.append('selected_symbols', selectedSymbols.join(','));
                        
                        const response = await fetch('/highlight-paths', {{
                            method: 'POST',
                            body: formData
                        }});
                        
                        if (!response.ok) {{
                            throw new Error(`Failed to highlight paths: ${{response.statusText}}`);
                        }}
                        
                        const result = await response.json();
                        console.log('Highlight result:', result);
                        
                        // Update image with highlighted paths
                        if (result.highlighted_image) {{
                            img.src = result.highlighted_image + '?t=' + Date.now();
                            
                            // Show detailed success message
                            if (result.total_lines > 0) {{
                                const message = `✅ Highlighted ${{result.total_lines}} connecting lines for ${{result.connected_symbols.length}} symbols`;
                                alert(message);
                            }} else {{
                                alert('⚠️ No connecting lines found. Symbols may be isolated or detection needs adjustment.');
                            }}
                        }} else {{
                            alert('No connecting lines found between selected symbols');
                        }}
                        
                    }} catch (error) {{
                        console.error('Error highlighting paths:', error);
                        alert('Failed to highlight paths: ' + error.message);
                    }} finally {{
                        highlightBtn.innerHTML = originalText;
                        highlightBtn.disabled = false;
                    }}
                }}
                
                // Initialize when image loads
                img.onload = () => {{
                    const rect = img.getBoundingClientRect();
                    const scaleX = img.width / img.naturalWidth;
                    const scaleY = img.height / img.naturalHeight;
                    
                    // Create symbol grid
                    const symbolGrid = document.getElementById('symbol-grid');
                    symbolGrid.innerHTML = '';
                    
                    // Create tooltip element for hover display
                    let tooltip = document.createElement('div');
                    tooltip.className = 'symbol-tooltip';
                    tooltip.id = 'symbol-tooltip';
                    document.body.appendChild(tooltip);
                    
                    // Create bounding boxes and symbol cards
                    detections.forEach(d => {{
                        // Create bounding box overlay
                        let box = document.createElement('div');
                        box.className = 'symbol-box';
                        box.dataset.symbolId = d.id;
                        box.style.position = 'absolute';
                        box.style.left = (d.bbox[0] * scaleX) + 'px';
                        box.style.top = (d.bbox[1] * scaleY) + 'px';
                        box.style.width = ((d.bbox[2] - d.bbox[0]) * scaleX) + 'px';
                        box.style.height = ((d.bbox[3] - d.bbox[1]) * scaleY) + 'px';
                        box.style.border = '3px solid rgb(' + d.color[0] + ',' + d.color[1] + ',' + d.color[2] + ')';
                        
                        // Add hover events for tooltip display
                        box.addEventListener('mouseenter', function(e) {{
                            const tagName = `${{d.id}} - ${{d.class}} (${{(d.confidence * 100).toFixed(1)}}%)`;
                            tooltip.textContent = tagName;
                            tooltip.classList.add('visible');
                            
                            // Position tooltip above the symbol
                            const boxRect = box.getBoundingClientRect();
                            const containerRect = container.getBoundingClientRect();
                            tooltip.style.left = (boxRect.left + boxRect.width/2 - tooltip.offsetWidth/2) + 'px';
                            tooltip.style.top = (boxRect.top - tooltip.offsetHeight - 10) + 'px';
                        }});
                        
                        box.addEventListener('mouseleave', function() {{
                            tooltip.classList.remove('visible');
                        }});
                        
                        box.addEventListener('mousemove', function(e) {{
                            // Update tooltip position to follow mouse
                            const containerRect = container.getBoundingClientRect();
                            tooltip.style.left = (e.clientX - tooltip.offsetWidth/2) + 'px';
                            tooltip.style.top = (e.clientY - tooltip.offsetHeight - 15) + 'px';
                        }});
                        
                        box.addEventListener('click', () => toggleSymbolSelection(d.id));
                        container.appendChild(box);
                        
                        // Create symbol card
                        let card = document.createElement('div');
                        card.className = 'symbol-card';
                        card.dataset.symbolId = d.id;
                        card.innerHTML = `
                            <div class="symbol-title">${{d.id}}</div>
                            <div class="symbol-info">
                                Class: ${{d.class}}<br>
                                Confidence: ${{(d.confidence * 100).toFixed(1)}}%<br>
                                Position: [${{d.bbox[0]}}, ${{d.bbox[1]}}, ${{d.bbox[2]}}, ${{d.bbox[3]}}]
                            </div>
                        `;
                        card.addEventListener('click', () => toggleSymbolSelection(d.id));
                        symbolGrid.appendChild(card);
                    }});
                    
                    updateUI();
                }};
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/graph")
async def graph(file: UploadFile = File(...)):
    """
    Build and return connection graph from detected symbols and lines.
    Graph represents symbol connectivity for path analysis.
    """
    try:
        validate_file(file)
        image_bytes = await file.read()
        _, detections, merged_line_segments = process_image(image_bytes)
        
        # Filter symbols only (exclude lines)
        symbols = [d for d in detections if "line" not in d["class"].lower()]
        
        # Build connection graph
        connection_graph, symbol_centers = build_connection_graph(symbols, merged_line_segments)
        
        return JSONResponse(content={
            "graph": connection_graph,
            "symbol_count": len(symbols),
            "line_count": len(merged_line_segments),
            "connections": sum(len(connections) for connections in connection_graph.values()) // 2
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Graph endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/highlight-paths")
async def highlight_paths(
    session_id: str = Form(...),
    selected_symbols: str = Form(...)
):
    """
    Highlight all paths connecting selected symbols.
    Uses cached file data from the session.
    """
    try:
        # Parse selected symbols
        selected_symbols_list = [s.strip() for s in selected_symbols.split(",") if s.strip()]
        if not selected_symbols_list:
            raise HTTPException(status_code=400, detail="No symbols selected")
        
        logger.info(f"Processing highlight request for symbols: {selected_symbols_list}")
        
        # Get cached file data
        if session_id not in uploaded_files_cache:
            raise HTTPException(status_code=400, detail="Session expired. Please upload the image again.")
        
        file_data = uploaded_files_cache[session_id]
        image_bytes = file_data['data']
        
        # Process the image
        image_filename, detections, merged_line_segments = process_image(image_bytes)
        
        # Filter symbols only
        symbols = [d for d in detections if "line" not in d["class"].lower()]
        
        logger.info(f"Found {len(symbols)} symbols and {len(merged_line_segments)} merged line segments")
        
        # Build connection graph and get symbol centers
        graph, symbol_centers = build_connection_graph(symbols, merged_line_segments)
        
        # Validate selected symbols exist
        valid_symbols = [s for s in selected_symbols_list if s in symbol_centers]
        if not valid_symbols:
            raise HTTPException(status_code=400, detail="Selected symbols not found in image")
        
        logger.info(f"Valid symbols: {valid_symbols}")
        
        # Find lines that connect to ANY of the selected symbols
        connecting_lines = []
        
        for line in merged_line_segments:
            connected_symbols = []
            
            # Check which symbols (selected or not) this line connects to
            for symbol in symbols:
                symbol_id = symbol["id"]
                x1, y1, x2, y2 = symbol["bbox"]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Check distance to symbol center
                distance = line.distance_to_point(cx, cy)
                
                # Check bounding box intersection with padding
                padding = 20
                intersects = line_intersects_rectangle(line, 
                                                     x1 - padding, y1 - padding, 
                                                     x2 + padding, y2 + padding)
                
                if distance <= Config.CONNECTION_THRESHOLD or intersects:
                    connected_symbols.append(symbol_id)
            
            # Include line if it connects to at least one selected symbol
            has_selected_symbol = any(sym in valid_symbols for sym in connected_symbols)
            
            if has_selected_symbol and len(connected_symbols) >= 1:
                connecting_lines.append(line)
                logger.info(f"Line connects: {connected_symbols}")
        
        logger.info(f"Found {len(connecting_lines)} connecting lines")
        
        # If still no connections, try even more relaxed approach
        if not connecting_lines:
            logger.info("No connections found, using relaxed distance threshold...")
            for line in merged_line_segments:
                for symbol_id in valid_symbols:
                    if symbol_id in symbol_centers:
                        cx, cy = symbol_centers[symbol_id]
                        # Use much larger threshold
                        if line.distance_to_point(cx, cy) <= Config.CONNECTION_THRESHOLD * 3:
                            connecting_lines.append(line)
                            logger.info(f"Added line near {symbol_id} with relaxed threshold")
                            break
        
        # Load the processed image for highlighting
        original_image_path = os.path.join(Config.UPLOAD_FOLDER, image_filename)
        img = cv2.imread(original_image_path)
        if img is None:
            raise HTTPException(status_code=500, detail="Could not load processed image")
        
        # Draw ALL detected lines in blue first (for debugging when no connections found)
        if len(connecting_lines) == 0:
            logger.info("No connecting lines found, drawing all detected lines in blue for debugging")
            for line in merged_line_segments:
                cv2.line(img, (int(line.x1), int(line.y1)), (int(line.x2), int(line.y2)), 
                        (255, 100, 0), 2)  # Blue lines for debugging
        
        # Draw connecting lines in red
        for line in connecting_lines:
            # Draw thick red line for highlighted paths
            cv2.line(
                img, 
                (int(line.x1), int(line.y1)), 
                (int(line.x2), int(line.y2)), 
                (0, 0, 255),  # Bright red color (BGR format)
                Config.LINE_THICKNESS + 2
            )
            
            # Draw yellow endpoints
            cv2.circle(img, (int(line.x1), int(line.y1)), 8, (0, 255, 255), -1)
            cv2.circle(img, (int(line.x2), int(line.y2)), 8, (0, 255, 255), -1)
        
        # Highlight selected symbols with green circles
        for symbol_id in valid_symbols:
            if symbol_id in symbol_centers:
                cx, cy = symbol_centers[symbol_id]
                # Draw thick green circle
                cv2.circle(img, (int(cx), int(cy)), 30, (0, 255, 0), 6)
                # Add text label
                cv2.putText(img, "SELECTED", (int(cx-45), int(cy-35)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
        
        # Save highlighted image
        highlighted_filename = f"highlighted_{uuid.uuid4().hex}.png"
        highlighted_path = os.path.join(Config.UPLOAD_FOLDER, highlighted_filename)
        cv2.imwrite(highlighted_path, img)
        
        logger.info(f"Saved highlighted image: {highlighted_filename}")
        
        # Prepare response data
        connected_lines_data = [
            [int(line.x1), int(line.y1), int(line.x2), int(line.y2)] 
            for line in connecting_lines
        ]
        
        return JSONResponse(content={
            "success": True,
            "connected_symbols": valid_symbols,
            "selected_symbols": valid_symbols,
            "connected_lines": connected_lines_data,
            "highlighted_image": f"/static/{highlighted_filename}",
            "total_lines": len(connecting_lines),
            "processing_info": {
                "total_symbols": len(symbols),
                "total_detections": len(detections),
                "selected_count": len(valid_symbols),
                "merged_line_segments": len(merged_line_segments),
                "graph_connections": len(graph)
            },
            "debug_info": {
                "symbol_centers": {k: list(v) for k, v in symbol_centers.items()},
                "connection_threshold": Config.CONNECTION_THRESHOLD,
                "all_line_count": len(merged_line_segments)
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Highlight paths error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring system status"""
    try:
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "upload_folder_exists": os.path.exists(Config.UPLOAD_FOLDER),
            "config": {
                "max_file_size_mb": Config.MAX_FILE_SIZE // (1024 * 1024),
                "connection_threshold": Config.CONNECTION_THRESHOLD,
                "confidence_threshold": Config.CONFIDENCE_THRESHOLD,
                "line_thickness": Config.LINE_THICKNESS
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"status": "unhealthy", "error": str(e)}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )

