import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

mode = 'KPCN'
recon_kernel_size = 9   #21
L = 9
kernel_size = 5
hidden_channels = 50    #100

permutation = [0, 3, 1, 2]

class KPCNN(nn.Module):
    def __init__(self, input_channels, layerCount = L):
        super(KPCNN, self).__init__()
        layers = [
            nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=kernel_size//2),
            nn.ReLU()
        ]
        for l in range(layerCount-2):
            layers += [
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2),
                nn.ReLU()
            ]
            #params = sum(p.numel() for p in layers[-2].parameters() if p.requires_grad)
            #print(params)

        out_channels = recon_kernel_size ** 2
        layers += [nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=kernel_size//2)]

        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)

        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits, dim=1)
        #return self.net(x)
