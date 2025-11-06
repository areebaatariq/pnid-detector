from ultralytics import YOLO
import os
import yaml


dataset_path = '/home/augmentation.gpu@vaival.tech/vaivaltech/hensis_pnid'

data_yaml = os.path.join(dataset_path, 'data.yaml')
project_dir = os.path.join(dataset_path, 'runs/train')
experiment_name = 'PID_YOLOv8'

num_classes = 109
model = YOLO('yolov8m.pt')

last_ckpt = os.path.join(project_dir, experiment_name, 'weights', 'last.pt')
resume_flag = False
if os.path.exists(last_ckpt):
    print(f"Resuming from checkpoint: {last_ckpt}")
    model = YOLO(last_ckpt)
    resume_flag = True

model.train(
    data=data_yaml,
    epochs=200,
    imgsz=1280,
    batch=4,
    device=0,
    augment=True,
    cache=True,
    project=project_dir,
    name=experiment_name,
    exist_ok=True,
    optimizer='Adam',
    lr0=0.003,
    patience=30,
    resume=resume_flag
)
