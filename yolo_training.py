from ultralytics import YOLO

# Load YOLOv11-nano
model = YOLO("yolo11n.pt")

# Train the model
model.train(
    data="PCB_dataset.yaml",
    epochs=120,        # <200 required
    batch=8,
    imgsz=900,
    name="pcb_detector_v1"
)

print("Training complete. Check runs/detect/pcb_detector_v1/")
