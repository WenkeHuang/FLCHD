from collections import namedtuple
from typing import Literal

import torch
from lightning import LightningModule
from torch import nn

from chdmodule.modules.base_models import ModelFactory, OutputFeatExtractor
from chdmodule.modules.cpt_layers import CPTPredictor, CPTProjector
from chdmodule.modules.losses import MeanIoULoss
from chdmodule.modules.attention import CrossAttention
from chdmodule.utilities.enums import Task
from chdmodule.utilities.logging import get_logger
from chdmodule.utilities.misc import process_heatmaps, trunc_normal_
from chdmodule.utilities.types import StepOutput

class FLCPTProto(nn.Module):
    def __init__(self, base_model_name, cls_name, cpt_name, **kwargs):
        super().__init__()
        self.model = CLATProto(base_model_name, cls_name, cpt_name, **kwargs)
        self.name = base_model_name
    
    def forward(self, x):
        output = self.model.forward(x)
        return output
    
    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch, batch_idx)
    
    def features(self, x):
        with torch.no_grad():
            if x.shape[0] > 16:
                features_list = []
                batch_size = 16
                for i in range(0, x.shape[0], batch_size):
                    batch = x[i:i+batch_size]
                    feat_output = self.model.feat_extractor(batch)
                    features_list.append(feat_output.feat)
                return torch.cat(features_list, dim=0)
            else:
                feat_output = self.model.feat_extractor(x)
                return feat_output.feat

CHDImageDataItem = namedtuple(
    "CHDImageDataItem",
    [
        "image",
        "concept",
        "mask",
        "target",
        "hospital_id",
        "id",
        "img_path",
        "mask_path",
    ],
)

CLATOutput = namedtuple(
    "CLATOutput", ["logits", "logits_cpt", "heatmaps", "proto_distances"]
    )


class CPTEmbeddingPathway(nn.Module):
    def __init__(self, in_feat: int, hidden_dim: int):
        super().__init__()
        self.pathway = nn.Sequential(
            nn.LayerNorm(in_feat),
            nn.Linear(in_feat, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pathway(x)


class ProtoTypeConceptPredictor(nn.Module):
    def __init__(self, hidden_dim: int, n_cpt: int):
        super().__init__()
        self.pos_prototypes = nn.Parameter(torch.randn(n_cpt, hidden_dim))
        self.neg_prototypes = nn.Parameter(torch.randn(n_cpt, hidden_dim))
        
        trunc_normal_(self.pos_prototypes, std=0.02)
        trunc_normal_(self.neg_prototypes, std=0.02)
        
        self.n_cpt = n_cpt
        
    def forward(self, cpt_embeds: torch.Tensor) -> tuple:
        batch_size = cpt_embeds.size(0)
        
        pos_dists = torch.zeros(batch_size, self.n_cpt, device=cpt_embeds.device)
        neg_dists = torch.zeros(batch_size, self.n_cpt, device=cpt_embeds.device)
        
        for i in range(self.n_cpt):
            concept_embed = cpt_embeds[:, i, :]
            
            pos_dists[:, i] = torch.sum((concept_embed - self.pos_prototypes[i]) ** 2, dim=1)
            neg_dists[:, i] = torch.sum((concept_embed - self.neg_prototypes[i]) ** 2, dim=1)
        
        logits = torch.sigmoid(neg_dists - pos_dists)
        
        distances = {
            "pos_distances": pos_dists,
            "neg_distances": neg_dists
        }
        
        return logits, distances


class FCClassifier(nn.Module):
    def __init__(self, cpt_hidden_dim: int, n_cpt: int, n_cls: int):
        super().__init__()
        
        flattened_dim = cpt_hidden_dim * n_cpt
        
        self.classifier = nn.Sequential(
            nn.Flatten(),  
            nn.Linear(flattened_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_cls)
        )
    
    def forward(self, cpt_embeds: torch.Tensor) -> torch.Tensor:
        return self.classifier(cpt_embeds)


class CLATProto(LightningModule):
    def __init__(
        self,
        base_model_name: str,
        cls_name: list[str],
        cpt_name: list[str],
        task: str = "multiclass",
        cpt_pos_weight: list[float] = None,
        img_size: int = 224,
        base_model_pretrain: bool = True,
        cpt_hidden_dim: int = 512,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        if task != Task.MULTICLASS:
            raise ValueError(f"Only support multiclass task, but got {task}")
        
        self.cls_num = len(cls_name)
        self.cpt_num = len(cpt_name)
        self.task = task
        self.img_size = img_size
        self.cls_name = cls_name
        self.cpt_name = cpt_name
        self.cpt_pos_weight = cpt_pos_weight
        self.base_model_name = base_model_name
        self.base_model_pretrain = base_model_pretrain
        self.cpt_hidden_dim = cpt_hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        hf_model = ModelFactory[self.base_model_name]
        self.base_model = hf_model.get(
            num_classes=self.cls_num,
            pretrained=self.base_model_pretrain,
            base_model=True,
        )
        self.feat_extractor = hf_model.feat_extractor(self.base_model)
        feat_dim = hf_model.feat_dim
        
        self.cpt_pathways = nn.ModuleList([
            CPTEmbeddingPathway(feat_dim, cpt_hidden_dim) 
            for _ in range(self.cpt_num)
        ])
        
        self.proto_predictor = ProtoTypeConceptPredictor(cpt_hidden_dim, self.cpt_num)
        
        self.fc_classifier = FCClassifier(cpt_hidden_dim, self.cpt_num, self.cls_num)

        self.setup("fit")
        
    def setup(self, stage):
        pos_weight = (
            torch.Tensor(self.cpt_pos_weight).to(self.device)
            if self.cpt_pos_weight
            else None
        )
        self.loss_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_guide = MeanIoULoss()
        
        self.proto_dist_loss_weight = 1

    def forward(self, x: torch.Tensor):
        bs = x.size(0)
        featex_output: OutputFeatExtractor = self.feat_extractor(x, output_heatmap=True)
        
        base_feat = featex_output.feat
        
        cpt_embeds = torch.zeros(bs, self.cpt_num, self.cpt_hidden_dim, device=x.device)
        for i in range(self.cpt_num):
            cpt_embeds[:, i] = self.cpt_pathways[i](base_feat)
        
        cpt_logits, proto_distances = self.proto_predictor(cpt_embeds)
        
        logits = self.fc_classifier(cpt_embeds)
        
        output = CLATOutput(
            logits=logits,
            logits_cpt=cpt_logits,
            heatmaps=featex_output.heatmaps,
            proto_distances=proto_distances
        )
        return output

    def prototype_distance_loss(self, proto_distances, concepts):
        pos_distances = proto_distances["pos_distances"]
        neg_distances = proto_distances["neg_distances"]
        
        pos_sample_loss = torch.sum(pos_distances * concepts) / (torch.sum(concepts) + 1e-6)
        neg_sample_loss = torch.sum(neg_distances * (1 - concepts)) / (torch.sum(1 - concepts) + 1e-6)
        
        return pos_sample_loss + neg_sample_loss

    def shared_step(self, batch: CHDImageDataItem, batch_idx: int):
        images: torch.Tensor = batch.image
        targets: torch.Tensor = batch.target
        concepts: torch.Tensor = batch.concept
        masks: torch.Tensor = batch.mask

        output: CLATOutput = self(images)

        loss_cpt = self.loss_bce(output.logits_cpt, concepts)
        
        loss_cls = self.loss_ce(output.logits, targets)

        heatmaps = process_heatmaps(output.heatmaps, self.img_size)
        loss_guide = self.loss_guide(heatmaps, masks)

        loss = loss_cpt + loss_cls + loss_guide

        loss_dict = {
            "Concept": loss_cpt,
            "CLS": loss_cls,
            "Guide": loss_guide,
        }

        return StepOutput(loss, loss_dict, output._asdict())

    def training_step(self, batch: CHDImageDataItem, batch_idx: int = 0):
        step_output = self.shared_step(batch, batch_idx)
        bs = batch.image.size(0)

        self.log(
            "Loss/Train",
            step_output.loss,
            batch_size=bs,
            prog_bar=True,
            sync_dist=True,
        )
        loss_dict = step_output.loss_dict
        loss_dict = {f"Loss/{k}/Train": v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, batch_size=bs, sync_dist=True)

        return step_output.to_dict()

    def validation_step(self, batch: CHDImageDataItem, batch_idx: int = 0):
        step_output = self.shared_step(batch, batch_idx)
        bs = batch.image.size(0)

        self.log(
            "Loss/Val",
            step_output.loss,
            batch_size=bs,
            prog_bar=True,
            sync_dist=True,
        )
        loss_dict = step_output.loss_dict
        loss_dict = {f"Loss/{k}/Val": v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, batch_size=bs, sync_dist=True)

        return step_output.to_dict()

    def test_step(self, batch: CHDImageDataItem, batch_idx: int = 0):
        step_output = self.shared_step(batch, batch_idx)
        bs = batch.image.size(0)

        self.log(
            "Loss/Test",
            step_output.loss,
            batch_size=bs,
            prog_bar=True,
            sync_dist=True,
        )
        loss_dict = step_output.loss_dict
        loss_dict = {f"Loss/{k}/Test": v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, batch_size=bs, sync_dist=True)

        return step_output.to_dict()