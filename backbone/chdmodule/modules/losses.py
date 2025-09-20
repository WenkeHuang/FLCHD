import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MeanIoULoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, heatmap: torch.Tensor, mask: torch.Tensor):
        mask = convert_img_to_float(mask)

        epsilon = 1e-6
        intersection = (heatmap * mask).sum(dim=(1, 2))
        union = (heatmap + mask).sum(dim=(1, 2)) - intersection

        iou = (intersection + epsilon) / (union + epsilon)

        loss = 1 - iou.mean()

        return loss


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        if labels.dim() > 1:
            labels = labels.squeeze()
        if labels.dtype == torch.float:
            labels = (labels > 0.5).long()

        similarity = F.cosine_similarity(
            features.unsqueeze(1), features.unsqueeze(0), dim=2
        )
        similarity = similarity / self.temperature

        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

        identity_mask = torch.eye(pos_mask.shape[0]).bool().to(pos_mask.device)
        pos_mask[identity_mask] = 0

        exp_sim = torch.exp(similarity)
        denominator = exp_sim.sum(dim=1) - exp_sim.diag()
        numerator = (exp_sim * pos_mask).sum(dim=1)

        valid_mask = numerator > 0
        loss = torch.zeros_like(numerator)
        loss[valid_mask] = -torch.log(
            numerator[valid_mask] / denominator[valid_mask] + 1e-8
        )

        return loss.mean()


class DistSmoothLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features: torch.Tensor):
        batch_size = features.size(0)
        features_centered = features - features.mean(dim=0)
        cov = torch.mm(features_centered.t(), features_centered) / (batch_size - 1)

        loss = torch.norm(cov - torch.eye(cov.size(0)).to(cov.device))
        return loss


class GroupDRO(nn.Module):
    def __init__(self, num_hospitals: int, criterion: nn.Module):
        super().__init__()
        self.register_buffer("group_weights", torch.ones(num_hospitals))
        self.adjustment_rate = 0.01
        self.num_hospitals = num_hospitals
        self.criterion = criterion

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, hospital_ids: torch.Tensor
    ):
        group_losses = []
        for h_id in range(self.num_hospitals):
            mask = hospital_ids == h_id
            if mask.any():
                group_loss = self.criterion(logits[mask], labels[mask])
                group_losses.append(group_loss)
            else:
                group_losses.append(torch.tensor(0.0).to(logits.device))

        group_losses = torch.stack(group_losses)

        self.group_weights = self.group_weights * torch.exp(
            self.adjustment_rate * group_losses.detach()
        )
        self.group_weights = self.group_weights / self.group_weights.sum()

        weighted_loss = (group_losses * self.group_weights).sum()
        return weighted_loss


class FeatureDRO(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features: torch.Tensor, hospital_ids: torch.Tensor):
        hospital_means = []

        for h_id in torch.unique(hospital_ids):
            mask = hospital_ids == h_id
            h_features = features[mask]
            hospital_means.append(h_features.mean(0))

        feature_diffs = []
        for i in range(len(hospital_means)):
            for j in range(i + 1, len(hospital_means)):
                diff = torch.norm(hospital_means[i] - hospital_means[j])
                feature_diffs.append(diff)

        return (
            torch.stack(feature_diffs).mean()
            if feature_diffs
            else torch.tensor(0.0).to(features.device)
        )


def convert_img_to_float(img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == torch.uint8:
        return img.float() / 255.0
    return img
