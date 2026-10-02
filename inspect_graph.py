import onnx

model = onnx.load("mobilenetv2-12.onnx")

print("Number of operators:", len(model.graph.node))

print("\nFirst 20 operators:")

for i, node in enumerate(model.graph.node[:20]):
    print(i, node.op_type, "->", node.output)
