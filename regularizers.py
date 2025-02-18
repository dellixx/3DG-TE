from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass

class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]


class Lambda3(Regularizer):
    def __init__(self, weight: float):
        super(Lambda3, self).__init__()
        self.weight = weight

    def forward(self, factor):
        ddiff = factor[1:] - factor[:-1]
        rank = int(ddiff.shape[1] / 2)
        diff = torch.sqrt(ddiff[:, :rank]**2 + ddiff[:, rank:]**2)**3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)

class Linear3(Regularizer):
    def __init__(self, weight: float):
        super(Linear3, self).__init__()
        self.weight = weight

    def forward(self, factor, W):
        rank = int(factor.shape[1] / 2)
        ddiff = factor[1:] - factor[:-1] - W.weight[:rank*2].t()
        diff = torch.sqrt(ddiff[:, :rank]**2 + ddiff[:, rank:]**2)**3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)


class Spiral3(Regularizer):
    def __init__(self, weight: float):
        super(Spiral3, self).__init__()
        self.weight = weight
        self.target_value =1.0

    def forward(self, factor, time_phase):
        
        
        
        ddiff = factor[1:] - factor[:-1] 
        ddiff_pahse = time_phase[1:] - time_phase[:-1]
        
        diff_d = time_phase - self.target_value  # (batch_size, rank)
        
        rank = int(ddiff.shape[1] / 2)
        rank1 = int(ddiff_pahse.shape[1] / 2)
        
        diff1 = torch.sqrt(ddiff[:, :rank]**2 + ddiff[:, rank:]**2)**3
        diff2 = torch.sqrt(ddiff_pahse[:, :rank1]**2 + ddiff_pahse[:, rank1:]**2)**3
        diff3 = torch.sum(torch.sqrt(diff_d**2)**3)
        
        x1 = torch.sum(diff1) / (factor.shape[0] - 1)
        x2 = torch.sum(diff2) / (time_phase.shape[0] - 1)
        x3 = diff3 / (time_phase.shape[0])
        
        return self.weight * x1 