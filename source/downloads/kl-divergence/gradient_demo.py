"""Autograd example from the KL article. Requires PyTorch; CPU float64."""
import torch

p = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
P = torch.stack((p, 1 - p))
Q = torch.tensor([0.5, 0.5], dtype=torch.float64)
u = P.log() - Q.log()

exact = (P * u).sum()            # 概率权重也参与求导
k1 = (P.detach() * u).sum()      # 模拟已采样后固定的频率
k2 = (P.detach() * u.square() / 2).sum()
k3 = (P.detach() * (torch.expm1(-u) + u)).sum()

for name, loss in [("exact", exact), ("k1", k1), ("k2", k2), ("k3", k3)]:
    grad, = torch.autograd.grad(loss, p, retain_graph=True)
    print(name, round(grad.item(), 6))
