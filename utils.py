import os
import gdown

def download_model_from_drive(file_id, output_path):
    """Download model file from Google Drive if it doesn't exist"""
    if os.path.exists(output_path):
        print(f"Model already exists at {output_path}")
        return
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)
    print(f"✅ Model downloaded to: {output_path}")
