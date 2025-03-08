from ultralytics import YOLO
from sahi.predict import predict

# Load a pretrained YOLOv8n model
model = YOLO("runs/detect/train18/weights/best.pt")
models = "runs/detect/train18/weights/best.pt"
# Define path to the image file
source = "tiles"

# Run inference on the source

#model.predict("C:/Users/user/Documents/solution/pythonProject7/tiles", save=True, imgsz=256,iou=0.1, conf=0.01,show_labels=False)
predict(
    model_type="yolov8",
    model_path=models,
    model_device="cpu",  # or 'cuda:0'
    model_confidence_threshold=0.01,
    source="C:/Users/user/Documents/solution/pythonProject7/shai/layeriamge.tif",
    slice_height=128,  # Smaller slice height
    slice_width=128,   # Smaller slice width
    visual_bbox_thickness=1,
    overlap_height_ratio=0,  # Increased overlap height ratio
    overlap_width_ratio=0,   # Increased overlap width ratio
    visual_hide_labels=True,  # Show labels for better debugging
)