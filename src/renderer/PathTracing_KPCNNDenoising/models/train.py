from dataSet import KPCNNDataset
import model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import glob
import time

eps = 0.00316

def crop_like(data, like, debug=False):
  if data.shape[-2:] != like.shape[-2:]:
    with torch.no_grad():
      dx, dy = data.shape[-2] - like.shape[-2], data.shape[-1] - like.shape[-1]
      data = data[:,:,dx//2:-dx//2,dy//2:-dy//2]
      if debug:
        print(dx, dy)
        print("After crop:", data.shape)
  return data
'''
def apply_kernel(weights, data):
    # apply softmax to kernel weights
    weights = weights.permute((0, 2, 3, 1)).to(device)
    _, _, h, w = data.size()
    weights = F.softmax(weights, dim=3).view(-1, w * h, model.recon_kernel_size, model.recon_kernel_size)

    # now we have to apply kernels to every pixel
    r = model.recon_kernel_size // 2
    data = F.pad(data[:,:3,:,:], (r,) * 4, "reflect")   # first pad the input
    
    # make slices
    R = []
    G = []
    B = []
    kernels = []
    for i in range(h):
      for j in range(w):
        pos = i * w + j
        ws = weights[:,pos:pos+1,:,:]
        kernels += [ws, ws, ws]
        sy, ey = i+r-r, i+r+r+1
        sx, ex = j+r-r, j+r+r+1
        R.append(data[:,0:1,sy:ey,sx:ex])
        G.append(data[:,1:2,sy:ey,sx:ex])
        B.append(data[:,2:3,sy:ey,sx:ex])
        #slices.append(data[:,:,sy:ey,sx:ex])
        
    reds = (torch.cat(R, dim=1).to(device)*weights).sum(2).sum(2)
    greens = (torch.cat(G, dim=1).to(device)*weights).sum(2).sum(2)
    blues = (torch.cat(B, dim=1).to(device)*weights).sum(2).sum(2)
    
    res = torch.cat((reds, greens, blues), dim=1).view(-1, 3, h, w).to(device)
    
    return res
'''
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

torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

trainDataSet = KPCNNDataset("dataSet/train")
dataloader = torch.utils.data.DataLoader(trainDataSet, batch_size=10, shuffle=True, num_workers=0)

diffuseNet = model.KPCNN(trainDataSet[0]['input_diff'].shape[-1]).to(device)
specularNet = model.KPCNN(trainDataSet[0]['input_spec'].shape[-1]).to(device)

print(diffuseNet, "CUDA:", next(diffuseNet.parameters()).is_cuda)
print(specularNet, "CUDA:", next(specularNet.parameters()).is_cuda)

loss_function = nn.L1Loss()
optimizer_diff = optim.Adam(diffuseNet.parameters(), lr=0.0001)
optimizer_spec = optim.Adam(specularNet.parameters(), lr=0.0001)

save_path_diff = "weights/KPCNN_diff_Weights"
save_path_spec = "weights/KPCNN_spec_Weights"
best_loss = 1000000.0
start = time.time()
for epoch in range(200):
    t1 = time.time()
    diffuseNet.train()
    specularNet.train()
    total_loss_diff = 0.0
    total_loss_spec = 0.0
    total_loss_final = 0.0

    for step, sample_batched in enumerate(dataloader):
        sample_diff = sample_batched['input_diff'].permute(model.permutation).to(device)  #N C H W
        sample_spec = sample_batched['input_spec'].permute(model.permutation).to(device)
        sample_albedo = sample_batched['albedo'].permute(model.permutation).to(device)
        gt_diff = sample_batched['diff_ref'].permute(model.permutation).to(device)
        gt_spec = sample_batched['spec_ref'].permute(model.permutation).to(device)
        gt_final = sample_batched['gtColor'].permute(model.permutation).to(device)

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
        # ---------- Final ----------
        sample_final = filtered_diff + torch.exp(filtered_spec) - 1.0
        loss_final = loss_function(sample_final, gt_final)
        total_loss_final += loss_final.item()

    avg_loss_final = total_loss_final / len(dataloader)
    print(f'Epoch {epoch+1:3d} | Loss Diff: {total_loss_diff/len(dataloader):.4f} '
          f'Spec: {total_loss_spec/len(dataloader):.4f} Final: {avg_loss_final:.4f}')
    if avg_loss_final < best_loss:
        best_loss = avg_loss_final
        torch.save(diffuseNet.state_dict(), save_path_diff)
        torch.save(specularNet.state_dict(), save_path_spec)

    print()
    print(time.perf_counter() - t1)
print('Finished Training')
print('Took', time.time() - start, 'seconds.')
