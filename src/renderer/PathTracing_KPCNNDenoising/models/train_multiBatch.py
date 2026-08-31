import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import model  # 你的模型文件
from dataSet import KPCNNDataset  # 你的数据集类

# -------------------- 配置 --------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

trainSetPath = 'C:/Users/fangzanbo/Desktop/FzbRenderer_nvvk/src/renderer/PathTracing_KPCNNDenoising/vulkanDataSet/'

# 原始 40 个场景的路径
sampleFolders = [trainSetPath + f'staircase_32_{i}_{j}' for i in range(1) for j in range(10)]
gtFolders = [trainSetPath + f'staircase_8192_{i}_0' for i in range(1) for _ in range(10)]

# 超参数
batch_size = 10
epochs_per_batch = 20
total_rounds = 10          # 总共循环多少轮（每轮训完所有场景一次）
scenes_per_batch = 3        # 每批 3 个场景
save_interval = 5           # 每隔多少 batch 保存一次 checkpoint

eps = 0.00316
checkpoint_path = "checkpoint.pth"
save_path_diff = "weights/KPCNN_diff_Weights"
save_path_spec = "weights/KPCNN_spec_Weights"
os.makedirs("weights", exist_ok=True)

# -------------------- 工具函数 --------------------
def apply_kernel(kernels, patch):
    N, C, H, W = patch.shape
    K2 = kernels.shape[1]
    K = int(K2 ** 0.5)
    assert K * K == K2, "kernels.shape[1] must be a perfect square"
    r = K // 2

    kernels = kernels.permute(0, 2, 3, 1).contiguous().view(N, H * W, K * K)
    neighborhood = F.unfold(
        F.pad(patch, (r, r, r, r), mode='constant', value=0.0),
        kernel_size=K
    )
    neighborhood = neighborhood.view(N, C, K * K, H * W)
    neighborhood = neighborhood.permute(0, 1, 3, 2)
    kernels = kernels.unsqueeze(1)
    out = (neighborhood * kernels).sum(dim=-1)
    return out.view(N, C, H, W)

def build_dataloader(sample_folders, gt_folders):
    """为给定的场景列表构建 DataLoader，并释放原始数据集"""
    dataset = KPCNNDataset(sample_folders, gt_folders)

    # 将所有样本转为 torch.Tensor 并堆叠（内存占用约 1–2 GB）
    all_input_diff, all_input_spec, all_albedo = [], [], []
    all_diff_ref, all_spec_ref, all_gtColor = [], [], []
    for s in dataset.samples:
        all_input_diff.append(torch.from_numpy(s['input_diff'].copy()).permute(2, 0, 1))
        all_input_spec.append(torch.from_numpy(s['input_spec'].copy()).permute(2, 0, 1))
        all_albedo.append(torch.from_numpy(s['albedo'].copy()).permute(2, 0, 1))
        all_diff_ref.append(torch.from_numpy(s['diff_ref'].copy()).permute(2, 0, 1))
        all_spec_ref.append(torch.from_numpy(s['spec_ref'].copy()).permute(2, 0, 1))
        all_gtColor.append(torch.from_numpy(s['gtColor'].copy()).permute(2, 0, 1))

    # 释放 KPCNNDataset 中的 numpy 数组
    del dataset

    tensor_data = TensorDataset(
        torch.stack(all_input_diff),
        torch.stack(all_input_spec),
        torch.stack(all_albedo),
        torch.stack(all_diff_ref),
        torch.stack(all_spec_ref),
        torch.stack(all_gtColor)
    )
    loader = DataLoader(tensor_data, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)
    return loader


# -------------------- 初始化模型 & 优化器 --------------------
# 先用一个临时数据集获取输入通道数
temp_dataset = KPCNNDataset(sampleFolders[:1], gtFolders[:1])
diff_channels = temp_dataset[0]['input_diff'].shape[-1]
spec_channels = temp_dataset[0]['input_spec'].shape[-1]
del temp_dataset

diffuseNet = model.KPCNN(diff_channels).to(device)
specularNet = model.KPCNN(spec_channels).to(device)
loss_function = nn.L1Loss()
optimizer_diff = optim.Adam(diffuseNet.parameters(), lr=0.0001)
optimizer_spec = optim.Adam(specularNet.parameters(), lr=0.0001)

# 尝试恢复 checkpoint
start_round = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    diffuseNet.load_state_dict(checkpoint['model_diff'])
    specularNet.load_state_dict(checkpoint['model_spec'])
    optimizer_diff.load_state_dict(checkpoint['optimizer_diff'])
    optimizer_spec.load_state_dict(checkpoint['optimizer_spec'])
    start_round = checkpoint.get('round', 0)
    print(f"已恢复 round {start_round}，继续训练...")
else:
    print("没有找到 checkpoint，从头训练。")

# -------------------- 分批训练 --------------------
all_scenes = list(zip(sampleFolders, gtFolders))
random.shuffle(all_scenes)   # 打乱场景顺序，避免顺序偏差

for round_idx in range(start_round, total_rounds):
    print(f"\n===== Round {round_idx + 1}/{total_rounds} =====")
    # 将场景列表分成每批 scenes_per_batch 个
    for batch_idx in range(0, len(all_scenes), scenes_per_batch):
        batch_scenes = all_scenes[batch_idx:batch_idx + scenes_per_batch]
        batch_sample_folders = [s[0] for s in batch_scenes]
        batch_gt_folders     = [s[1] for s in batch_scenes]

        print(f"  加载场景批次 {batch_idx//scenes_per_batch + 1}，场景数: {len(batch_scenes)}")
        train_loader = build_dataloader(batch_sample_folders, batch_gt_folders)

        # 对当前批次训练 epochs_per_batch 轮
        for epoch in range(1, epochs_per_batch + 1):
            diffuseNet.train()
            specularNet.train()
            total_loss_diff, total_loss_spec, total_loss_final = 0.0, 0.0, 0.0

            for step, (sample_diff, sample_spec, sample_albedo,
                       gt_diff, gt_spec, gt_final) in enumerate(train_loader):
                sample_diff  = sample_diff.to(device)
                sample_spec  = sample_spec.to(device)
                sample_albedo = sample_albedo.to(device)
                gt_diff      = gt_diff.to(device)
                gt_spec      = gt_spec.to(device)
                gt_final     = gt_final.to(device)

                # ---------- Diffuse ----------
                optimizer_diff.zero_grad()
                kernel_diff = diffuseNet(sample_diff)
                filtered_diff = apply_kernel(kernel_diff, sample_diff[:, :3, :, :])
                filtered_diff = filtered_diff * (sample_albedo + eps)
                loss_diff = loss_function(filtered_diff, gt_diff)
                loss_diff.backward()
                optimizer_diff.step()
                total_loss_diff += loss_diff.item()

                # ---------- Specular ----------
                optimizer_spec.zero_grad()
                kernel_spec = specularNet(sample_spec)
                filtered_spec = apply_kernel(kernel_spec, sample_spec[:, :3, :, :])
                loss_spec = loss_function(filtered_spec, gt_spec)
                loss_spec.backward()
                optimizer_spec.step()
                total_loss_spec += loss_spec.item()

                # ---------- Final (no grad) ----------
                with torch.no_grad():
                    sample_final = filtered_diff + torch.exp(filtered_spec) - 1.0
                    loss_final = loss_function(sample_final, gt_final)
                    total_loss_final += loss_final.item()

            avg_diff = total_loss_diff / len(train_loader)
            avg_spec = total_loss_spec / len(train_loader)
            avg_final = total_loss_final / len(train_loader)
            print(f"    Batch [{batch_idx//scenes_per_batch+1}] "
                  f"Epoch {epoch:2d}/{epochs_per_batch} | "
                  f"Diff: {avg_diff:.4f} Spec: {avg_spec:.4f} Final: {avg_final:.4f}")

        # 释放当前批次数据
        del train_loader

    # 每个 round 结束保存 checkpoint（包含优化器状态）
    torch.save({
        'round': round_idx + 1,
        'model_diff': diffuseNet.state_dict(),
        'model_spec': specularNet.state_dict(),
        'optimizer_diff': optimizer_diff.state_dict(),
        'optimizer_spec': optimizer_spec.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint 已保存 (round {round_idx + 1})")

# 训练完成后单独保存模型权重
torch.save(diffuseNet.state_dict(), save_path_diff)
torch.save(specularNet.state_dict(), save_path_spec)
print("训练完成！最终权重已保存。")