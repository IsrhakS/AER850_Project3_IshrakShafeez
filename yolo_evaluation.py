from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/pcb_detector_v1/weights/best.pt")

# Run prediction on 3 test images
model.predict(
    source="Evaluation/",
    imgsz=900,
    conf=0.25,
    save=True
)

print("Evaluation complete")
