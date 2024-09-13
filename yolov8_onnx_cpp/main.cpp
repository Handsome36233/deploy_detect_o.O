#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <memory>

using namespace std;

typedef struct _DL_RESULT
{
    int classId;
    float confidence;
    cv::Rect box;
    vector<cv::Point2f> keyPoints;
} DL_RESULT;

void PrintSessionInfo(const Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator) {
    // 获取输入输出信息
    auto input_count = session.GetInputCount();
    auto output_count = session.GetOutputCount();

    cout << "Input Count: " << input_count << endl;
    cout << "Output Count: " << output_count << endl;

    // 打印输入名称和形状
    for (int i = 0; i < input_count; ++i) {
        string input_name = session.GetInputNameAllocated(i, allocator).get();
        vector<int64_t> input_shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

        cout << "Input " << i << " Name: " << input_name << endl;
        cout << "Input " << i << " Shape: ";
        for (const auto& dim : input_shape) cout << dim << ' ';
        cout << endl;
    }

    // 打印输出名称和形状
    for (int i = 0; i < output_count; ++i) {
        string output_name = session.GetOutputNameAllocated(i, allocator).get();
        vector<int64_t> output_shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

        cout << "Output " << i << " Name: " << output_name << endl;
        cout << "Output " << i << " Shape: ";
        for (const auto& dim : output_shape) cout << dim << ' ';
        cout << endl;
    }
}

void BlobFromImage(cv::Mat& iImg, float* iBlob) {
    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;

    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < imgHeight; h++)
        {
            for (int w = 0; w < imgWidth; w++)
            {
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = static_cast<float>(
                        (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <onnx_model_path> <image_path>\n";
        return -1;
    }
    const char* model_path = (const char *)argv[1];
    const char* image_path = (const char *)argv[2];
    const char* save_path = "output.jpg";   // 保存结果
    
    // 配置参数
    float rectConfidenceThreshold = 0.3;
    float iouThreshold = 0.5;
    int img_width = 640;
    int img_height = 384;

    // 初始化 ONNX Runtime 环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolov8_inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // 加载模型
    Ort::Session session(env, model_path, session_options);
    Ort::AllocatorWithDefaultOptions allocator;
    // 打印输入输出信息
    PrintSessionInfo(session, allocator);
    Ort::RunOptions options;
    vector<const char*> inputNodeNames;
    vector<const char*> outputNodeNames;

    size_t inputNodesNum = session.GetInputCount();
    for (size_t i = 0; i < inputNodesNum; i++)
    {
        Ort::AllocatedStringPtr input_node_name = session.GetInputNameAllocated(i, allocator);
        char* temp_buf = new char[50];
        strcpy(temp_buf, input_node_name.get());
        inputNodeNames.push_back(temp_buf);
    }
    size_t OutputNodesNum = session.GetOutputCount();
    for (size_t i = 0; i < OutputNodesNum; i++)
    {
        Ort::AllocatedStringPtr output_node_name = session.GetOutputNameAllocated(i, allocator);
        char* temp_buf = new char[10];
        strcpy(temp_buf, output_node_name.get());
        outputNodeNames.push_back(temp_buf);
    }
    options = Ort::RunOptions{ nullptr };

    // 处理图片
    cv::Mat image = cv::imread(image_path);
    cv::Mat show_img = image.clone();
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    float resizeScalesW = image.cols / float(img_width);
    float resizeScalesH = image.rows / float(img_height);
    cv::resize(image, image, cv::Size(img_width, img_height));
    // 转tensor
    float* blob = new float[image.total() * 3];
    BlobFromImage(image, blob);
    vector<int64_t> inputNodeDims = { 1, 3, img_height, img_width };
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * img_width * img_height,
        inputNodeDims.data(), inputNodeDims.size());
    // 模型推理
    auto outputTensor = session.Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),
        outputNodeNames.size());
    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    vector<int64_t> outputNodeDims = tensor_info.GetShape();
    // 取出输出数据
    auto output = outputTensor.front().GetTensorMutableData<float>();

    int signalResultNum = outputNodeDims[1];
    int strideNum = outputNodeDims[2];
    cout << "signalResultNum: " << signalResultNum << endl;
    cout << "strideNum: " << strideNum << endl;
    vector<int> class_ids;
    vector<float> confidences;
    vector<cv::Rect> boxes;
    for (int i = 0; i < strideNum; ++i)
    {
        float tmp_class_score[signalResultNum-4];
        for (int j = 0; j < signalResultNum-4; ++j) tmp_class_score[j] = output[strideNum*(j+4)+i];
        auto classesScores = max_element(tmp_class_score, tmp_class_score + signalResultNum-4);
        auto classId = distance(tmp_class_score, classesScores);
        
        if (*classesScores > rectConfidenceThreshold)
        {
            confidences.push_back(*classesScores);
            class_ids.push_back(classId);
            float x = output[strideNum*0+i];
            float y = output[strideNum*1+i];
            float w = output[strideNum*2+i];
            float h = output[strideNum*3+i];

            int left = int((x - 0.5 * w) * resizeScalesW);
            int top = int((y - 0.5 * h) * resizeScalesH);

            int width = int(w * resizeScalesW);
            int height = int(h * resizeScalesH);

            boxes.push_back(cv::Rect(left, top, width, height));
        }
    }
    vector<int> nmsResult;
    cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);
    vector<DL_RESULT> oResult;
    for (int i = 0; i < nmsResult.size(); ++i)
    {
        int idx = nmsResult[i];
        DL_RESULT result;
        result.classId = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        oResult.push_back(result);
    }
    cout << "oResult: " << oResult.size() << endl;
    for (auto& re : oResult) {
        cv::rectangle(show_img, re.box, (0, 255, 0), 3);
    }
    cv::imwrite(save_path, show_img);
    cout << "save result to " << save_path << endl;
    return 0;
}
