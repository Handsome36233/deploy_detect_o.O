#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <array>
#include <numeric>
#include <functional>
#include <fstream>

using namespace std;

typedef struct _DL_RESULT
{
    int classId;
    float confidence;
    cv::Rect box;
    vector<cv::Point2f> keyPoints;
} DL_RESULT;

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

void transpose(float* input, float* output, const int* m_n, int* pose) {
    // 计算新数组的大小
    int totalSize = 1;
    for (int k = 0; k < 4; ++k) {
        totalSize *= m_n[pose[k]];
    }

    int m_n_n[8];
    for (int k = 0; k < 4; k++)
        m_n_n[k] = m_n[k];
    for (int k = 4; k < 8; k++)
        m_n_n[k] = 1;

    int i[4], s[4];
    for (i[0] = 0;  i[0] < m_n[pose[0]]; ++i[0])
        for (i[1] = 0; i[1] < m_n[pose[1]]; ++i[1])
            for (i[2] = 0; i[2] < m_n[pose[2]]; ++i[2])
                for (i[3] = 0; i[3] < m_n[pose[3]]; ++i[3]) {
                    int old_index = 0;
                    for (int j = 0; j < 4; j++) {
                        s[j] = i[j];
                        for (int k = 0; k < 3; k++)
                            s[j] = s[j] * m_n_n[pose[j] + k + 1];
                        old_index += s[j];
                    }

                    int new_index = 0;
                    for (int j = 0; j < 4; j++) {
                        s[j] = i[j];
                        for (int k = j + 1; k < 4; k++)
                            s[j] = s[j] * m_n_n[pose[k]];
                        new_index += s[j];
                    }
                    output[new_index] = input[old_index];
                }
}

int argmax(const float* input, int length) {
    if (length == 0) {
        return -1;
    }
    int max_index = 0;
    float max_value = input[0];
    for (int i = 1; i < length; ++i) {
        if (input[i] > max_value) {
            max_value = input[i];
            max_index = i;
        }
    }
    return max_index;
}

void softmax(float* input, float* output, int length) {
    float max_input = *std::max_element(input, input + length);
    float sum_exp = 0.0f;
    for (int i = 0; i < length; ++i) {
        output[i] = std::exp(input[i] - max_input);
        sum_exp += output[i];
    }
    for (int i = 0; i < length; ++i) {
        output[i] /= sum_exp;
    }
}

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

std::tuple<cv::Mat, std::pair<float, float>, std::pair<float, float>> letterbox(const cv::Mat& im, const cv::Size& new_shape = cv::Size(640, 640), const cv::Scalar& color = cv::Scalar(0, 0, 0)) {
    // 获取当前图像的宽高
    int img_h = im.rows;
    int img_w = im.cols;

    // 计算缩放比例
    float r_w = static_cast<float>(new_shape.width) / img_w;
    float r_h = static_cast<float>(new_shape.height) / img_h;
    float r = std::min(r_w, r_h);  // 选择最小比例

    // 计算新的宽高
    int new_w = static_cast<int>(std::round(img_w * r));
    int new_h = static_cast<int>(std::round(img_h * r));
    std::pair<int, int> new_unpad = {new_w, new_h};

    std::pair<float, float> ratio = {new_w / float(new_shape.width), new_h / float(new_shape.height)};
    // 计算左右和上下的填充
    int dw = new_shape.width - new_w;
    int dh = new_shape.height - new_h;

    float dw_half = dw / 2.0f;  // 将填充分配到两侧
    float dh_half = dh / 2.0f;

    // 如果新尺寸和原图尺寸不一致，进行缩放
    cv::Mat resized;
    if (cv::Size(new_w, new_h) != cv::Size(img_w, img_h)) {
        cv::resize(im, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = im;
    }

    // 计算填充的上下左右值
    int top = static_cast<int>(std::round(dh_half - 0.1));
    int bottom = static_cast<int>(std::round(dh_half + 0.1));
    int left = static_cast<int>(std::round(dw_half - 0.1));
    int right = static_cast<int>(std::round(dw_half + 0.1));

    // 进行填充
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT, color);

    // 返回填充后的图像、缩放比例和填充尺寸
    return {padded, ratio, {dw_half, dh_half}};
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

void saveFloatArrayToFile(const float* array, size_t size, const std::string& filename) {
    std::ofstream outFile(filename); // 打开文件
    if (!outFile) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < size; ++i) {
        outFile << array[i] << "\n"; // 写入每个值并换行
    }
    outFile.close(); // 关闭文件
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
    int anchors[9][2] = {
        {26, 26}, {56, 16}, {116, 19}, {235, 23}, {540, 32},
        {139, 139}, {413, 64}, {364, 292}, {479, 703}
    };

    // 初始化 ONNX Runtime 环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolov5_inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // 加载模型
    Ort::Session session(env, model_path, session_options);
    auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    int img_height = input_shape[2];
    int img_width = input_shape[3];
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
    // cv::resize(image, image, cv::Size(img_width, img_height));
    cv::Size new_shape = cv::Size(img_width, img_height);
    auto [padded_image, ratio, padding] = letterbox(image, new_shape, cv::Scalar(114, 114, 114));
    std::cout << "ratio: " << ratio.first << " " << ratio.second << " padding: " << padding.first << " " << padding.second << std::endl;
    // 转tensor
    cv::imwrite("padded_image.jpg", padded_image);
    float* blob = new float[padded_image.total() * 3];
    BlobFromImage(padded_image, blob);
    vector<int64_t> inputNodeDims = { 1, 3, img_height, img_width };
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * img_width * img_height,
        inputNodeDims.data(), inputNodeDims.size());
    // 模型推理
    auto outputTensor = session.Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),
        outputNodeNames.size());
    
    vector<int> class_ids;
    vector<float> confidences;
    vector<cv::Rect> boxes;

    auto output_count = session.GetOutputCount();
    for (int i = 0; i < output_count; ++i) {
        Ort::TypeInfo typeInfo = outputTensor[i].GetTypeInfo();
        auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
        vector<int64_t> outputNodeDims = tensor_info.GetShape();
        auto output = outputTensor[i].GetTensorMutableData<float>();

        vector<int64_t> output_shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        cout<<"output_shape: "<<output_shape[0]<<" "<<output_shape[1]<<" "<<output_shape[2]<<" "<<output_shape[3]<<endl;
        int shape[4] = {3, int(output_shape[1]/3), int(output_shape[2]), int(output_shape[3])};
        cout<<"shape: "<<shape[0]<<" "<<shape[1]<<" "<<shape[2]<<" "<<shape[3]<<endl;
        int pose[4] = {2, 3, 0, 1};
        int class_num = int(output_shape[1]/3)-5;
        int total_size = output_shape[0]*output_shape[1]*output_shape[2]*output_shape[3];
        float transposed_output[total_size];
        memset(transposed_output, 0.0, total_size * sizeof(float));
        transpose(output, transposed_output, shape, pose);
        // if (i < 2) continue;
        // saveFloatArrayToFile(transposed_output, total_size, "output.txt");
        for (int h = 0; h < output_shape[2]; ++h) {
            for (int w = 0; w < output_shape[3]; ++w) {
                for (int k = 0; k < 3; ++k) {
                    int curr = (h*output_shape[3]+w)*3*(class_num+5)+k*(class_num+5);
                    float box_confidence = sigmoid(transposed_output[curr+4]);
                    float prob_score[class_num];
                    memset(prob_score, 0.0, class_num * sizeof(float));
                    softmax(transposed_output+curr+5, prob_score, class_num);
                    int class_id = argmax(prob_score, class_num);
                    if (box_confidence*prob_score[class_id] > rectConfidenceThreshold){
                        confidences.push_back(box_confidence);
                        class_ids.push_back(class_id);
                        float x = sigmoid(transposed_output[curr]) * 2 - 0.5 + w;
                        float y = sigmoid(transposed_output[curr+1]) * 2 - 0.5 + h;
                        x *= (img_width / output_shape[3]);
                        y *= (img_height / output_shape[2]);
                        float w = pow(sigmoid(transposed_output[curr+2])*2, 2)*anchors[i*3+k][0];
                        float h = pow(sigmoid(transposed_output[curr+3])*2, 2)*anchors[i*3+k][1];

                        int left = int((x - 0.5 * w));
                        int top = int((y - 0.5 * h));

                        // 还原到原图坐标
                        left = int((left - padding.first) / ratio.first);  // x1
                        top = int((top - padding.second) / ratio.second);  // y1
                        w = int(w / ratio.first);  // w
                        h = int(h / ratio.second);  // h

                        left *= resizeScalesW;
                        top *= resizeScalesH;
                        w *= resizeScalesW;
                        h *= resizeScalesH;

                        boxes.push_back(cv::Rect(left, top, w, h));
                    }
                }
            }
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
    for (auto& re : oResult) {
        cv::rectangle(show_img, re.box, (0, 255, 0), 3);
        std::cout << "classId: " << re.classId << " confidence: " << re.confidence << " box: " << re.box << std::endl;
    }
    cv::imwrite(save_path, show_img);
    cout << "save result to " << save_path << endl;
    return 0;
}
