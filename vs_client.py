import requests

BASE_URL = "https://twanda-hyenine-sharice.ngrok-free.dev"
# 🔹 Detect symbols
file_path = "/home/areeba.tariq@vaival.tech/Desktop/hensis-model-api/P & ID. KPC Process Unit 5-202 - LP Slug Catcher. LP Slug Catcher & LP Gas Heater_page-0001.jpg"
with open(file_path, "rb") as f:
    response = requests.post(f"{BASE_URL}/detect_tiled", files={"file": f})

if response.status_code == 200:
    data = response.json()
    print("Detections count:", data["count"])
    for det in data["detections"]:
        print(det)
else:
    print("Error:", response.text)

# 🔹 Visualize and get annotated image
with open(file_path, "rb") as f:
    response = requests.post(f"{BASE_URL}/visualize", files={"file": f})

if response.status_code == 200:
    with open("annotated_result.png", "wb") as out:
        out.write(response.content)
    print("✅ Annotated image saved as annotated_result.png")
else:
    print("Error:", response.text)
