import dataSet
import model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

eval_data = dataSet.preprocess_input("dataSet/eval/eval3.exr", "dataSet/eval/evalRef3.exr")
input_channelCount = eval_data['input_diff'].shape[-1]
diffuseNet = model.KPCNN(input_channelCount).to(device)
specularNet = model.KPCNN(input_channelCount).to(device)

weights_path = "weights/KPCNN_diff_Weights"
assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
diffuseNet.load_state_dict(torch.load(weights_path))
diffuseNet.eval()

weights_path = "weights/KPCNN_spec_Weights"
assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
specularNet.load_state_dict(torch.load(weights_path))
specularNet.eval()

dummy_diff = torch.randn(1, input_channelCount, 256, 256, device=device)
dummy_spec = torch.randn(1, input_channelCount, 256, 256, device=device)

dynamic_axes = {
    'input':  {0: 'batch', 2: 'height', 3: 'width'},
    'output': {0: 'batch', 2: 'height', 3: 'width'}
}

torch.onnx.export(
    diffuseNet,
    dummy_diff,
    "kpcnn_diffuse.onnx",
    input_names=['input'],
    output_names=['kernel_weights'],
    dynamic_axes=dynamic_axes,
    opset_version=17,
    do_constant_folding=True
)
torch.onnx.export(
    specularNet,
    dummy_spec,
    "kpcnn_specular.onnx",
    input_names=['input'],
    output_names=['kernel_weights'],
    dynamic_axes=dynamic_axes,
    opset_version=17,
    do_constant_folding=True
)

import onnx
onnx_model = onnx.load("kpcnn_diffuse.onnx")
onnx.checker.check_model(onnx_model)
print("diff ONNX model is valid!")

onnx_model = onnx.load("kpcnn_specular.onnx")
onnx.checker.check_model(onnx_model)
print("spec ONNX model is valid!")

print("ONNX 导出完成")