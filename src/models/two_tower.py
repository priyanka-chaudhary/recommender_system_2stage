from __future__ import annotations
import torch
from torch import nn

class TwoTower(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.02)
        nn.init.normal_(self.item_emb.weight, std=0.02)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_idx)
        v = self.item_emb(item_idx)
        return (u * v).sum(dim=-1)

    def user_vectors(self) -> torch.Tensor:
        return self.user_emb.weight.detach()

    def item_vectors(self) -> torch.Tensor:
        return self.item_emb.weight.detach()
