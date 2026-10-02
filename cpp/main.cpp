#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>
#include <vector>

int main()
{
    const int num_threads = 8;

    const char* model_path = "../mobilenetv2-12.onnx";
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DL_Inference");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(num_threads);

    Ort::Session session(env, model_path, session_options);

    std::vector<int64_t> input_shape = {1, 3, 224, 224};

    size_t input_size = 1 * 3 * 224 * 224;

    cv::Mat image = cv::imread("../Image.jpg");
    if (image.empty())
    {
        std::cerr << "Failed to load image\n";
        return 1;
    }
    cv::resize(image, image, cv::Size(224, 224));
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);

    std::vector<float> input_data(input_size);
    int c = 0,h = 0,w = 0;
    for (c = 0; c < 3; ++c)
    {
        for (h = 0; h < 224; ++h)
        {
            for (w = 0; w < 224; ++w)
            {
                input_data[c * 224 * 224 + h * 224 + w] = image.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    Ort::MemoryInfo memory_info(
        "Cpu",
        OrtAllocatorType::OrtArenaAllocator,
        0,
        OrtMemTypeDefault
    );

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(),
        input_data.size(),
        input_shape.data(),
        input_shape.size()
    );

    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name_allocated = session.GetInputNameAllocated(0, allocator);
    auto output_name_allocated = session.GetOutputNameAllocated(0, allocator);

    const char* input_name = input_name_allocated.get();
    const char* output_name = output_name_allocated.get();

    for (int i = 0; i < 10; ++i)
    {
        session.Run(
            Ort::RunOptions{nullptr},
            &input_name,
            &input_tensor,
            1,
            &output_name,
            1
        );
    }

    const int iterations = 100;

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < iterations; ++i)
    {
        auto output = session.Run(
            Ort::RunOptions{nullptr},
            &input_name,
            &input_tensor,
            1,
            &output_name,
            1
        );
    }

    auto end = std::chrono::steady_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

    double average_ms = total_ms / iterations;

    auto output = session.Run(
        Ort::RunOptions{nullptr},
        &input_name,
        &input_tensor,
        1,
        &output_name,
        1
    );

    float* output_data = output[0].GetTensorMutableData<float>();

    int predicted_class = 0;

    for (int i = 1; i < 1000; ++i)
    {
        if (output_data[i] > output_data[predicted_class])
        {
            predicted_class = i;
        }
    }

    std::cout << "Input shape: [1, 3, 224, 224]\n";
    std::cout << "Output shape: [1, 1000]\n";
    std::cout << "Threads: " << num_threads << "\n";
    std::cout << "Average inference latency: "<< average_ms << " ms\n";
    std::cout << "Predicted class index: " << predicted_class << "\n";
    
    return 0;
}
