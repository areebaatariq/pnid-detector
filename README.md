# P&ID Symbol Detection & Path Highlighting System

A powerful FastAPI application for detecting P&ID (Piping and Instrumentation Diagram) symbols using YOLOv8 deep learning, with intelligent path highlighting and connection graph analysis.

## Features

- **YOLOv8 Object Detection** - State-of-the-art deep learning model for accurate P&ID symbol recognition
- **Multi-Format Support** - Supports JPEG, PNG, BMP, and TIFF image formats (max 20MB)
- **Smart Line Processing** - Advanced algorithms merge collinear line segments and detect complex symbol connections
- **Interactive Web Interface** - Click and select multiple symbols to highlight connecting paths
- **Connection Graph Analysis** - Build graphs to analyze symbol relationships and identify flow paths
- **RESTful API** - Clean API endpoints for integration with other systems
- **Real-time Processing** - Fast inference with configurable confidence thresholds

## Tech Stack

- **Backend**: FastAPI, Python 3.8+
- **ML Model**: YOLOv8 (Ultralytics)
- **Image Processing**: OpenCV, NumPy
- **Deep Learning**: PyTorch

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/pnid-vision.git
cd pnid-vision
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Ensure the model file exists**
   - The `best.pt` YOLOv8 model file should be in the project root directory

4. **Run the server**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start at `http://localhost:8000`

### Using Docker

```bash
docker build -t pnid-vision .
docker run -p 8000:8000 pnid-vision
```

## API Endpoints

### `GET /` - Home Page
Interactive web interface for uploading and analyzing P&ID images.

### `POST /detect` - Symbol Detection
Detect symbols in a P&ID image and return an interactive interface.

**Request:**
- `file`: Image file (JPEG, PNG, BMP, TIFF)

**Response:** Interactive HTML page with:
- Detected symbols with bounding boxes
- Click-to-select functionality
- Symbol statistics

**Example using cURL:**
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_pnid_image.png"
```

**Example using Python:**
```python
import requests

files = {'file': open('your_pnid_image.png', 'rb')}
response = requests.post('http://localhost:8000/detect', files=files)
print(response.text)  # Returns interactive HTML
```

### `POST /graph` - Connection Graph
Build and return a connection graph from detected symbols and lines.

**Request:**
- `file`: Image file

**Response:**
```json
{
  "graph": {
    "gate valve-01": ["pipe-01", "pump-01"],
    "pump-01": ["gate valve-01", "tank-01"]
  },
  "symbol_count": 15,
  "line_count": 23,
  "connections": 42
}
```

**Example:**
```python
import requests

files = {'file': open('your_pnid_image.png', 'rb')}
response = requests.post('http://localhost:8000/graph', files=files)
graph_data = response.json()
print(f"Found {graph_data['connections']} connections")
```

### `POST /highlight-paths` - Path Highlighting
Highlight all paths connecting selected symbols.

**Request:**
- `session_id`: Session ID from /detect endpoint
- `selected_symbols`: Comma-separated list of symbol IDs

**Response:**
```json
{
  "success": true,
  "connected_symbols": ["gate valve-01", "pump-01"],
  "connected_lines": [[100, 200, 300, 200], [300, 200, 400, 300]],
  "highlighted_image": "/static/highlighted_abc123.png",
  "total_lines": 5
}
```

### `GET /health` - Health Check
Check system status and configuration.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "upload_folder_exists": true,
  "config": {
    "max_file_size_mb": 20,
    "connection_threshold": 30,
    "confidence_threshold": 0.4
  }
}
```

## Configuration

The application can be configured via the `Config` class in `main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_PATH` | `best.pt` | Path to YOLOv8 model file |
| `UPLOAD_FOLDER` | `static` | Folder for processed images |
| `MAX_FILE_SIZE` | 20MB | Maximum upload file size |
| `ALLOWED_EXTENSIONS` | jpg, jpeg, png, bmp, tiff, tif | Supported image formats |
| `CONNECTION_THRESHOLD` | 30 | Distance threshold for symbol connections |
| `CONFIDENCE_THRESHOLD` | 0.4 | Minimum detection confidence |
| `LINE_THICKNESS` | 4 | Thickness of highlighted lines |

## Project Structure

```
pnid-vision/
├── main.py              # FastAPI application
├── best.pt              # YOLOv8 trained model
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
├── README.md            # This file
├── data.yaml            # YOLO training configuration
├── static/              # Processed images folder
└── .github/
    └── workflows/       # CI/CD configuration
```

## Usage Examples

### Web Interface
1. Navigate to `http://localhost:8000`
2. Upload a P&ID image
3. View detected symbols with bounding boxes
4. Click symbols to select them
5. Click "Highlight Paths" to visualize connections

### Programmatic Usage

```python
import requests

# 1. Detect symbols
with open('pnid_diagram.png', 'rb') as f:
    response = requests.post('http://localhost:8000/detect', files={'file': f})

# 2. Get connection graph
with open('pnid_diagram.png', 'rb') as f:
    response = requests.post('http://localhost:8000/graph', files={'file': f})
    graph = response.json()
    
print(f"Detected {graph['symbol_count']} symbols")
print(f"Found {graph['connections']} connections")

# 3. Analyze connections
for symbol, connections in graph['graph'].items():
    print(f"{symbol} connects to: {', '.join(connections)}")
```

## Supported P&ID Symbols

The model detects various P&ID symbols including:
- Valves (gate, ball, check, control, etc.)
- Pumps and compressors
- Tanks and vessels
- Instruments and sensors
- Piping and connections
- And many more...

## API Documentation

Interactive API documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the object detection framework
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [OpenCV](https://opencv.org/) for image processing
