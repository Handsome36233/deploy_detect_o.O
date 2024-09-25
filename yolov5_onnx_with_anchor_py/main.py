import cv2
import onnxruntime

import detect_utils
from detect_utils import detect

# yolo 配置
detect_utils.OBJ_THRESH = 0.6
detect_utils.NMS_THRESH = 0.25
size = (544, 960)
detect_onnx = "./checkpoints/pc_yolov5s.onnx"
image_path = "./data/9.jpg"
anchors = [[26, 26], [56, 16], [116, 19], [235, 23], [540, 32],
           [139, 139], [413, 64], [364, 292], [479, 703]]
detect_session = onnxruntime.InferenceSession(detect_onnx)

image = cv2.imread(image_path)
H, W, _ = image.shape
boxes, labels, scores = detect(image, detect_session, anchors, size=size)
for box, _ in zip(boxes, labels):
    xmin, ymin, xmax, ymax = box
    xmin = int(xmin * W / size[0])
    ymin = int(ymin * H / size[1])
    xmax = int(xmax * W / size[0])
    ymax = int(ymax * H / size[1])
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

image = cv2.resize(image, (W // 2, H // 2))
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
