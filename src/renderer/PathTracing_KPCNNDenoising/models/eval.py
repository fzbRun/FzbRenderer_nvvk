import dataSet
import model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import os
import json

eps = 0.00316
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def unsqueeze_all(d):
  for k, v in d.items():
    d[k] = torch.unsqueeze(v, dim=0)
  return d
def apply_kernel(kernels, patch):
    N, C, H, W = patch.shape
    K2 = kernels.shape[1]
    K = int(K2 ** 0.5)
    assert K * K == K2, "kernels.shape[1] must be a perfect square"
    r = K // 2

    # [N, K*K, H, W] -> [N, H, W, K*K] -> [N, H*W, K*K]
    kernels = kernels.permute(0, 2, 3, 1).contiguous().view(N, H * W, K * K)

    # 对每个像素自己的 K*K 权重归一化
    kernels = F.softmax(kernels, dim=-1)  # [N, H*W, K*K]

    # [N, C, H, W] -> unfold -> [N, C*K*K, H*W]
    neighborhood = F.unfold(
        F.pad(patch, (r, r, r, r), mode='reflect'),
        kernel_size=K
    )

    # [N, C, K*K, H*W]
    neighborhood = neighborhood.view(N, C, K * K, H * W)

    # 调整为 [N, C, H*W, K*K]，与 kernels 对齐
    neighborhood = neighborhood.permute(0, 1, 3, 2)

    # kernels: [N, 1, H*W, K*K]
    kernels = kernels.unsqueeze(1)

    out = (neighborhood * kernels).sum(dim=-1)  # [N, C, H*W]
    return out.view(N, C, H, W)


def denoise(diffuseNet, specularNet, data, debug=False):
    with torch.no_grad():
        loss_function = nn.L1Loss()

        data = dataSet.to_torch_tensors(data)
        if len(data['input_diff'].size()) != 4:
            for k, v in data.items(): data[k] = v.unsqueeze(0)

        for k in data: data[k] = data[k].to(device)

        # ---------- Diffuse ----------
        input_diff = data['input_diff'].permute(model.permutation)
        gt_diff = data['diff_ref'].permute(model.permutation)
        kernel_diff = diffuseNet(input_diff)
        filtered_diff = apply_kernel(kernel_diff, input_diff[:, :3, :, :])
        lossDiff = loss_function(filtered_diff, gt_diff).item()

        # ---------- Specular ----------
        input_spec = data['input_spec'].permute(model.permutation)
        gt_spec = data['spec_ref'].permute(model.permutation)
        kernel_spec = specularNet(input_spec)
        filtered_spec = apply_kernel(kernel_spec, input_spec[:, :3, :, :])
        lossSpec = loss_function(filtered_spec, gt_spec).item()

        # ---------- Final ----------
        albedo = data['albedo'].permute(model.permutation)
        filtered_color = filtered_diff * (albedo + eps) + torch.exp(filtered_spec) - 1.0

        print("Sample, denoised, gt")
        sz = 15

        sample_color = data['sampleColor'].cpu().numpy()[0, :]
        filtered_color_cpu = filtered_color.cpu().permute([0, 2, 3, 1]).numpy()[0, :]
        gt_color = data['gtColor'].cpu().numpy()[0, :]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        dataSet.show_data2(sample_color, normalize=True, ax=axes[0], show=False)
        axes[0].set_title('Sample')
        dataSet.show_data2(filtered_color_cpu, normalize=True, ax=axes[1], show=False)
        axes[1].set_title('Denoised')
        dataSet.show_data2(gt_color, normalize=True, ax=axes[2], show=False)
        axes[2].set_title('Ground Truth')

        plt.tight_layout()
        plt.show()

        # 计算最终损失时用 Tensor
        gt_color = data['gtColor'].permute(model.permutation)
        lossFinal = loss_function(filtered_color, gt_color).item()

        if debug:
            print("LossDiff:", lossDiff)
            print("LossSpec:", lossSpec)
            print("LossFinal:", lossFinal)
def main():
    eval_data = dataSet.preprocess_input("dataSet/eval/eval3.exr", "dataSet/eval/evalRef3.exr")
    eval_data = dataSet.crop(eval_data, (1280//2, 720//2), 300)

    diffuseNet = model.KPCNN(eval_data['input_diff'].shape[-1]).to(device)
    specularNet = model.KPCNN(eval_data['input_spec'].shape[-1]).to(device)

    weights_path = "weights/KPCNN_diff_Weights"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    diffuseNet.load_state_dict(torch.load(weights_path))
    diffuseNet.eval()

    weights_path = "weights/KPCNN_spec_Weights"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    specularNet.load_state_dict(torch.load(weights_path))
    specularNet.eval()

    denoise(diffuseNet, specularNet, eval_data, debug=True)

if __name__ == '__main__':
    main()