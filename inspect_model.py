import onnxruntime as ort

model_path = "mobilenetv2-12.onnx"

session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"]
)

print("Inputs:")
for inp in session.get_inputs():
    print("Name :", inp.name)
    print("Shape:", inp.shape)
    print("Type :", inp.type)

print("\nOutputs:")
for out in session.get_outputs():
    print("Name :", out.name)
    print("Shape:", out.shape)
    print("Type :", out.type)
