# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.nn import functional as F
from configs import config


class CrossEntropy(nn.Module):

    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label)

    def _forward(self, score, target):

        loss = self.criterion(score, target)

        return loss

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1: # 2
            score = [score]

        balance_weights = config.LOSS.BALANCE_WEIGHTS
        sb_weights = config.LOSS.SB_WEIGHTS
        if len(balance_weights) == len(score):
            return sum([
                w * self._forward(x, target)
                for (w, x) in zip(balance_weights, score)
            ])
        elif len(score) == 1:
            return sb_weights * self._forward(score[0], target)

        else:
            raise ValueError(
                "lengths of prediction and target are not identical!")


class OhemCrossEntropy(nn.Module):

    def __init__(self,
                 ignore_label=-1, # 255
                 thres=0.7, # 0.9
                 min_kept=100000, # 131072
                 weight=None): # [ 1.0023,0.9843, ]
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres # 0.9
        self.min_kept = max(1, min_kept) # 131072
        self.ignore_label = ignore_label # 255
        # weight: [ 1.0023,0.9843, ]
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def _ce_forward(self, score, target):
        """

        Args:
            score: (batch_size, 2, height, width)
            target: (batch_size, height, width)

        Returns:
            loss: (batch_size, height, width)

        """

        loss = self.criterion(score, target)
        return loss

    def _ohem_forward(self, score, target, **kwargs):
        """
        Args:
            score: (batch_size, 2, height, width)
            target: (batch_size, height, width)
                0, 1, 255 로만 이루어져 있음.
            **kwargs:

        Returns:

        """
        # (batch_size, height, width) -> (batch_size * height * width)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label  # 정상적인 것만 (_,)
        # mask.shape : (batch_size * height * width)

        ignore_removed_target = target.clone()  # (batch_size, height, width)
        ignore_removed_target[ignore_removed_target == self.ignore_label] = 0
        unsqueezed_ignore_removed_target = ignore_removed_target.unsqueeze(
            1)  # (batch_size, 1, height, width)
        pred = F.softmax(score, dim=1)  # (batch_size, 2,  height, width)
        """
pred 텐서에서 첫 번째 차원(인덱스 1)을 기준으로, 
    unsqueeze_target에서 추출한 인덱스에 해당하는 값을 가져옵니다.
즉, pred 텐서에서 unsqueeze_target에서 추출한 인덱스의 위치에 해당하는 값을 추출합니다.
        """
        # pred: (batch_size, height, width)
        # unsqueezed_ignore_removed_target: (batch_size, 1, height, width)
        pred = pred.gather(1, unsqueezed_ignore_removed_target)
        # pred:  (batch_size, 1, height, width)

        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        # pred.shape : batch_size * height * width -> mask True인 것만 예측값이 큰 순서로 정렬
        # ind.shape : batch_size * height * width -> mask True인 것만 예측값이 큰 순서로 정렬
        if mask.sum() == 0:
            return torch.tensor(0.).to(score.device)
        # min_kept: 131072
        min_value = pred[min(self.min_kept, pred.numel() - 1)]  # 여기까지
        # min_value: pred[min(맞춰야할 픽셀 수, 131072)]
        # self.thresh: 0.9
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, scores, target):
        """
        Args:
            scores:
                [x_extra_p_output, x_output] :
                    [(batch_size, 2, height, width), (batch_size, 2, height, width)]
                x_output
                    (batch_size, 2, height, width)

            target: [batch_size, height, width]
                0, 1, 255 로만 이루어져 있음.

        Returns:
            total_sum: (batch_size, height, width)
        """

        if not (isinstance(scores, list) or isinstance(scores, tuple)):
            scores = [scores]

        balance_weights = config.LOSS.BALANCE_WEIGHTS  # [0.4, 1.0]
        sb_weights = config.LOSS.SB_WEIGHTS  # 1.0
        loss_version = 1
        if len(balance_weights) == len(scores):
            # functions = [self._ce_forward, self._ohem_forward]
            functions = [self._ce_forward] * \
                        (len(balance_weights) - 1) + [self._ohem_forward]
            if loss_version == 0:
                return sum([
                    w * func(score, target)
                    for (w, score, func) in zip(balance_weights, scores, functions)
                ])
            else:
                total_sum = 0
                for idx, (weight, score, func) in enumerate(zip(balance_weights, scores, functions)):
                    # func: _ce_forward, _ohem_forward
                    a = func(score, target)
                    if idx == 0:
                        a = a.mean()
                    total_sum += weight * a
                return total_sum
        elif len(scores) == 1:
            return sb_weights * self._ohem_forward(scores[0], target)

        else:
            raise ValueError(
                "lengths of prediction and target are not identical!")


def weighted_bce(bd_pre, target):
    n, c, h, w = bd_pre.size()
    log_p = bd_pre.permute(0, 2, 3, 1).contiguous().view(1, -1)
    target_t = target.view(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    loss = F.binary_cross_entropy_with_logits(log_p,
                                              target_t,
                                              weight,
                                              reduction='mean')

    return loss


class BoundaryLoss(nn.Module):

    def __init__(self, coeff_bce=20.0):
        super(BoundaryLoss, self).__init__()
        self.coeff_bce = coeff_bce

    def forward(self, bd_pre, bd_gt):

        bce_loss = self.coeff_bce * weighted_bce(bd_pre, bd_gt)
        loss = bce_loss

        return loss


if __name__ == '__main__':
    a = torch.zeros(2, 64, 64)
    a[:, 5, :] = 1
    pre = torch.randn(2, 1, 16, 16)

    Loss_fc = BoundaryLoss()
    loss = Loss_fc(pre, a.to(torch.uint8))
