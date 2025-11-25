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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://hensis-dev.intellico.works",
        "https://hensis-pnid-dev.intellico.works",
        "https://hensis-nextjs.vercel.app"
    ],
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

# Detect endpoint with dictionary response and error handling
@app.post("/detect", response_class=HTMLResponse)
async def detect(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_filename, detections = process_image(image_bytes)
        html_image_path = f"/static/{image_filename}"

        # Prepare tag counts
        tag_counts = {cls: class_counters[cls] for cls in class_counters}

        # Keep the HTML exactly the same
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
                const detections = {json.dumps(detections)};

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

        # Success response
        response_data = {
            "status": True,
            "message": "Detection completed successfully",
            "data": [
                {
                    "html_content": html_content,
                    "tags_count": tag_counts
                }
            ],
            "errors": []
        }

    except Exception as e:
        # Error handling
        response_data = {
            "status": False,
            "message": "An error occurred during detection.",
            "data": [],
            "errors": [str(e)]  # Add the error message to the "errors" field
        }

    return HTMLResponse(content=json.dumps(response_data), media_type="application/json")







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
# def find_path_with_lines(graph, symbol_centers, line_segments, start_symbols, connection_threshold=25):
#     """Find connected path and return both symbols and the specific lines that connect them"""
#     # Use BFS to find all connected symbols
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
    
#     # Now find the specific lines that connect these symbols
#     path_lines = []
#     for line in line_segments:
#         connected_symbols_for_line = []
        
#         for symbol_id in visited_symbols:
#             cx, cy = symbol_centers[symbol_id]
#             if line.distance_to_point(cx, cy) <= connection_threshold:
#                 connected_symbols_for_line.append(symbol_id)
        
#         # If this line connects 2 or more symbols in our path, include it
#         if len(connected_symbols_for_line) >= 2:
#             path_lines.append(line)
    
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

#     # Draw merged lines on image (with increased width)
#     for line in merged_line_segments:
#         cv2.line(img, (int(line.x1), int(line.y1)), (int(line.x2), int(line.y2)), (0, 0, 255), 4)  # Increased width to 4

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
# @app.post("/highlight-paths")
# async def highlight_paths(file: UploadFile = File(...), selected_symbols: list = []):
#     if not selected_symbols:
#         return JSONResponse(content={"error": "No symbols selected"}, status_code=400)

#     image_bytes = await file.read()
#     _, detections, merged_line_segments = process_image(image_bytes)

#     # Separate symbols from lines
#     symbols = [d for d in detections if "line" not in d["class"].lower()]
    
#     # Build the improved connection graph for the symbols and lines
#     graph, symbol_centers = build_connection_graph(symbols, merged_line_segments)

#     # Find connected path with specific lines
#     connected_symbols, path_lines = find_path_with_lines(
#         graph, symbol_centers, merged_line_segments, selected_symbols
#     )

#     # Convert line segments back to coordinate format for frontend
#     connected_lines = [[int(line.x1), int(line.y1), int(line.x2), int(line.y2)] for line in path_lines]

#     # Highlight the selected paths in the image
#     img = cv2.imread(f"{UPLOAD_FOLDER}/{image_filename}")
#     for line in path_lines:
#         # Use a distinct color (e.g., red) for highlighted paths
#         cv2.line(img, (int(line.x1), int(line.y1)), (int(line.x2), int(line.y2)), (0, 0, 255), 4)  # Red color, thickness 4

#     # Save the image with highlighted paths
#     highlighted_filename = f"highlighted_{uuid.uuid4().hex}.png"
#     highlighted_image_path = os.path.join(UPLOAD_FOLDER, highlighted_filename)
#     cv2.imwrite(highlighted_image_path, img)

#     return JSONResponse(content={
#         "connected_symbols": connected_symbols,
#         "connected_lines": connected_lines,
#         "highlighted_image": f"/static/{highlighted_filename}"
#     })

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

