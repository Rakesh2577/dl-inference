# DL Inference

This is a simple hands-on project I built to understand how Deep Learning inference works and how CPU performance affects inference.

I used a pretrained MobileNetV2 ONNX model and ran inference using Python and C++ with ONNX Runtime.

## What I wanted to learn

* What happens during DL inference
* How input and output tensors work
* How ONNX Runtime runs a pretrained model
* How to measure inference latency
* How profiling helps identify expensive operators
* How CPU threads affect inference performance
* How this connects with C++ performance optimization

## Inference Flow

Image/Input
→ Preprocessing
→ Input Tensor
→ MobileNetV2
→ ONNX Runtime
→ Output Tensor

## Model

I used the pretrained MobileNetV2 model in ONNX format.

Input:

```text
[1, 3, 224, 224]
```

The input is a float32 tensor with:

* Batch = 1
* Channels = 3
* Height = 224
* Width = 224

Output:

```text
[1, 1000]
```

The output contains scores for 1000 classes.

## Python Inference

First I loaded the image and converted it into the format required by the model.I used OpenCV for image preprocessing so that the Python and C++ preprocessing are the same.

The preprocessing steps were:

* Resize image to 224 × 224
* Convert to RGB
* Convert to float32
* Normalize values between 0 and 1
* Change HWC format to CHW
* Add batch dimension

Then I passed the tensor to ONNX Runtime and got the output tensor.

The model predicted a class index from the output.

For my test image, the predicted class index was: 627

## Benchmarking

I measured only the model inference time using `session.run()`.

I used warm-up iterations before measuring the latency.

The Python inference latency was around:

```text
2.5 ms
```

The exact value can vary between runs.

## Profiling

I enabled ONNX Runtime profiling to understand which operators were taking more time.

The profile showed that convolution (`Conv`) kernels were among the major contributors to the execution time.

Some of the slower Conv kernel timings were around:

```text
0.8 ms
0.7 ms
0.6 ms
```

This helped me understand that the model is doing a lot of computation inside convolution operations.

## C++ Inference

I then implemented the inference part in C++ using ONNX Runtime.

I used OpenCV for image preprocessing and then created the input tensor in C++ and passed it to the same MobileNetV2 ONNX model.

The C++ program uses:

* C++17
* ONNX Runtime
* CMake
* CPU execution
* Intra-op threading

The C++ inference also produced the same predicted class:627

## Thread Experiment

I tested different ONNX Runtime intra-op thread counts.

| Threads | Average latency |
| ------: | --------------: |
|       1 |        7.93 ms |
|       2 |        4.49 ms |
|       4 |        2.97 ms |
|       8 |        2.57 ms |

Increasing the number of threads reduced the inference latency.

The improvement was not linear because there is other work involved, along with synchronization and memory-related overhead.

In my tests, 8 threads gave the lowest latency among the configurations I tested.

## What I learned

Through this project I understood the basic flow of DL inference from input tensor to output tensor.

I also understood that the Python code mainly controls the inference, while the actual heavy computation is performed by the runtime and its optimized kernels.

Profiling helped me see which operators were taking more time.

The threading experiment helped me understand how CPU parallelism can reduce inference latency, but adding more threads does not always give proportional speedup.

This also connected with what I already know about C++ performance optimization, such as multithreading, CPU computation and memory behavior.

## What I want to learn next

I want to learn more about how the optimized CPU kernels inside inference runtimes work, especially:

* SIMD and vectorization
* ARM NEON
* CPU cache and memory access
* Convolution optimization
* Profiling inference kernels
* How inference runtimes choose and execute optimized kernels

This project is a starting point for learning more about DL inference and C++ performance optimization.
