import torch
import onnx
# from onnx import OperatorExportTypes
from onnxsim import simplify
model = torch.jit.load("/home/fsw/Documents/codes/nanodet/mobilenetv2_nanodet.pt")

print(model.graph)

x = torch.randn((1, 3, 320, 320), requires_grad=True)
torch.onnx.export(model,
                  x,
                  "mobilenetv2_nanodet.onnx",
                  export_params=True,
                  verbose=True,
                  opset_version=13,
                  # do_constant_folding=True,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  # enable_onnx_checker=False,
                  )
onnx_model = onnx.load("mobilenetv2_nanodet.onnx")
onnx.checker.check_model(onnx_model)


