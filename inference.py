import time

import numpy as np
import onnxruntime as ort
import cv2

def preprocess(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

session_options = ort.SessionOptions()
#session_options.enable_profiling = True
session_options.intra_op_num_threads = 8

session = ort.InferenceSession(
    "mobilenetv2-12.onnx",
    sess_options=session_options,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
input_tensor = preprocess("Image.jpg")


for _ in range(10):
    session.run(None, {input_name: input_tensor})


iterations = 100

start = time.perf_counter()

for _ in range(iterations):
    session.run(None, {input_name: input_tensor})

end = time.perf_counter()

latency_ms = ((end - start) / iterations) * 1000

output = session.run(None, {input_name: input_tensor})
predicted_class = np.argmax(output[0][0])


print("Average inference latency:", latency_ms, "ms")
print("Predicted class index:", predicted_class)


#profile_file = session.end_profiling()

#print("Profile saved to:", profile_file)
