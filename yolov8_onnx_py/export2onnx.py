from ultralytics import YOLO

# Load a model
model = YOLO("/home/jinfan/Desktop/detect/checkpoints/best.pt")  # load an official model
# Set imgsz to 384, 640 for best performance
model.model.args["imgsz"] = [384, 640]
model.export(format="onnx")
