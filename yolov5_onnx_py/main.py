import cv2
import numpy as np
import onnxruntime as ort



# 预处理图像，调整大小并进行归一化
def preprocess_image(image):
    # 进行letterbox处理，保持宽高比，并进行填充
    img, ratio, (dw, dh) = letterbox(image, new_shape=IMG_SIZE)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
    img = np.ascontiguousarray(img)
    img = img.astype(np.float16)
    img /= 255.0  # 归一化到 [0, 1]
    img = np.expand_dims(img, axis=0)  # 添加batch维度
    return img, ratio, dw, dh

# 后处理推理输出
def postprocess_output(output, img_shape, ratio, dw, dh):
    boxes = output[0][:, :4]  # 边界框坐标
    scores = output[0][:, 4] * output[0][:, 5:].max(1)  # 置信度
    classes = output[0][:, 5:].argmax(1)  # 类别

    # 只保留置信度高于阈值的检测结果
    indices = np.where(scores > CONF_THRESHOLD)[0]
    boxes = boxes[indices]
    scores = scores[indices]
    classes = classes[indices]

    # 将 YOLOv5 格式 (cx, cy, w, h) 转换为 (x1, y1, x2, y2)
    boxes[:, 0] -= boxes[:, 2] / 2  # cx - w/2
    boxes[:, 1] -= boxes[:, 3] / 2  # cy - h/2
    boxes[:, 2] += boxes[:, 0]  # x1 + w
    boxes[:, 3] += boxes[:, 1]  # y1 + h

    # 还原到原始图像尺寸
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio[0]  # x1, x2
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio[1]  # y1, y2

    # 非极大值抑制（NMS）
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONF_THRESHOLD, IOU_THRESHOLD)
    if len(indices) > 0:
        indices = indices.flatten()
        boxes = boxes[indices]
        scores = scores[indices]
        classes = classes[indices]

    return boxes, scores, classes

# Letterbox function to resize image and maintain aspect ratio
def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # scale ratio
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

# 在原图上绘制边界框
def draw_boxes(image, boxes, scores, classes):
    for (box, score, class_id) in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f'{class_id}: {score:.2f}'
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 画框
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)  # 写标签
    return image

# 主函数，加载 ONNX 模型并推理
def run_inference(image_path, model_path):
    # 1. 加载图像
    image = cv2.imread(image_path)

    # 2. 加载 ONNX 模型
    session = ort.InferenceSession(model_path)

    # 3. 预处理图像
    input_image, ratio, dw, dh = preprocess_image(image)

    # 4. 模型推理
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_image})[0]

    # 5. 后处理输出
    boxes, scores, classes = postprocess_output(output, image.shape, ratio, dw, dh)

    # 6. 绘制边界框
    output_image = draw_boxes(image, boxes, scores, classes)

    # 7. 显示推理结果
    cv2.imshow('YOLOv5 Inference', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

IMG_SIZE = (640, 640)
CONF_THRESHOLD =0.25
IOU_THRESHOLD=0.45
image_path = './data/bus.jpg'
model_path = './checkpoints/yolov5s.onnx'
run_inference(image_path, model_path)
