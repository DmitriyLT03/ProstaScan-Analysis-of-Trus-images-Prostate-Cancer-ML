# from typing import Optional, List

# import torch
# import torch.nn.functional as F
# from torch.nn.modules.loss import _Loss


# def soft_jaccard_score(
#     output: torch.Tensor,
#     target: torch.Tensor,
#     smooth: float = 0.0,
#     eps: float = 1e-7,

# ) -> torch.Tensor:
#     intersection = torch.sum(output * target)
#     cardinality = torch.sum(output + target)

#     union = cardinality - intersection
#     jaccard_score = (intersection + smooth) / (union + smooth).clamp_min(eps)
#     return jaccard_score


# __all__ = ["JaccardLoss"]


# class JaccardLoss(_Loss):
#     def __init__(
#         self,
#         classes: Optional[List[int]] = None,
#         log_loss: bool = False,
#         from_logits: bool = True,
#         smooth: float = 0.0,
#         eps: float = 1e-7,
#     ):
#         super(JaccardLoss, self).__init__()

#         self.classes = classes
#         self.from_logits = from_logits
#         self.smooth = smooth
#         self.eps = eps
#         self.log_loss = log_loss

#     def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

#         y_pred = F.logsigmoid(y_pred.float()).exp()

#         bs = y_true.size(0)

#         y_true = y_true.view(bs, 1, -1)
#         y_pred = y_pred.view(bs, 1, -1)

#         scores = soft_jaccard_score(
#             y_pred,
#             y_true.type(y_pred.dtype),
#             smooth=self.smooth,
#             eps=self.eps,
#         )

#         if self.log_loss:
#             loss = -torch.log(scores.clamp_min(self.eps))
#         else:
#             loss = 1.0 - scores

#         return loss.mean()


import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc