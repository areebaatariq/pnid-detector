# PNID Symbol Detection API

This FastAPI application provides an API endpoint for detecting PNID (Piping and Instrumentation Diagram) symbols in images using YOLOv8.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

## Usage

The API provides a single endpoint `/detect` that accepts POST requests with image files.

### Endpoints

#### POST /detect

Accepts an image file and returns the detection results.

Parameters:
- `file`: The image file (PNG format)

Response Format:
The endpoint returns a JSON object containing:
- `png_base64`: Base64 encoded PNG image with bounding boxes
- `html`: Interactive HTML page with clickable bounding boxes
- `detections`: List of detected objects with their coordinates and unique IDs
- `image_info`: Image dimensions and detection statistics

Example using curl:
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.png" \
  -o response.json
```

Example using Python:
```python
import requests
import json
import base64

# Send request
files = {'file': open('your_image.png', 'rb')}
response = requests.post('http://localhost:8000/detect', files=files)
result = response.json()

# Save PNG image
png_data = base64.b64decode(result['png_base64'])
with open('result.png', 'wb') as f:
    f.write(png_data)

# Save HTML view
with open('result.html', 'w') as f:
    f.write(result['html'])

# Print detections
print("Detected objects:", len(result['detections']))
for detection in result['detections']:
    print(f"- {detection['id']} at coordinates {detection['bbox']}")
```

## Features

- Detects 109 different PNID symbols
- Draws bounding boxes with unique colors for each symbol type
- Interactive HTML view with clickable boxes to show labels
- Unique IDs for each detected instance (e.g., "gate valve-01", "gate valve-02")