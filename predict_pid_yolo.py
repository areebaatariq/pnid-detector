# from ultralytics import YOLO
# import os

# # Load your trained model
# model_path = "runs/train/PID_YOLOv8/weights/best.pt"
# model = YOLO(model_path)

# # Path to your test image
# image_path = "/home/augmentation.gpu@vaival.tech/Desktop/Cropped img3/3.6.png"

# # Run prediction
# results = model.predict(source=image_path, conf=0.4, save=True)

# print(f"\n✅ Prediction complete! Check results here:")
# print(results[0].save_dir)


import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/train/PID_YOLOv8/weights/best.pt")

# Path to your test image
image_path = "/home/augmentation.gpu@vaival.tech/Desktop/Cropped img3/3.6.png"

# Run prediction
results = model.predict(source=image_path, conf=0.4, hide_labels=True, hide_conf=True)

# Extract image and bounding boxes
img = results[0].orig_img.copy()
boxes = results[0].boxes  # xyxy, class, confidence

# Convert boxes to list of tuples
box_list = []
for box in boxes:
    xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
    cls = int(box.cls[0])         # class index
    label = results[0].names[cls] # class name
    box_list.append({"xyxy": xyxy, "label": label, "show_label": False})

# Mouse callback
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is inside any box
        for b in box_list:
            x1, y1, x2, y2 = map(int, b["xyxy"])
            if x1 <= x <= x2 and y1 <= y <= y2:
                b["show_label"] = not b["show_label"]  # toggle label

cv2.namedWindow("Prediction")
cv2.setMouseCallback("Prediction", on_mouse)

while True:
    display_img = img.copy()
    
    # Draw all boxes
    for b in box_list:
        x1, y1, x2, y2 = map(int, b["xyxy"])
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if b["show_label"]:
            cv2.putText(display_img, b["label"], (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Prediction", display_img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
