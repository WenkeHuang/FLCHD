import os
from collections import namedtuple
from enum import Enum
from functools import partial
from typing import Any, Protocol
from transformers import logging

import torch
from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
    PreTrainedModel,
    ResNetPreTrainedModel,
    SwinPreTrainedModel,
    ViTPreTrainedModel,
)

HF_CACHE_DIR = os.getenv("HF_CACHE_DIR")
HF_LOAD_KWARGS = {
    "cache_dir": HF_CACHE_DIR,
    "local_files_only": True,
    "trust_remote_code": True,
    "ignore_mismatched_sizes": True,
    "attn_implementation": "eager",
}


class FeatExtractFn(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        output_heatmap: bool = False,
        output_featmap: bool = False,
    ) -> Any: ...


OutputFeatExtractor = namedtuple("OutputFeatExtractor", ["feat", "featmap", "heatmaps"])


class ModelFactory(Enum):
    resnet18 = "microsoft/resnet-18"
    resnet50 = "microsoft/resnet-50"
    vit_small = "WinKawaks/vit-small-patch16-224"
    vit_base = "google/vit-base-patch16-224"
    swin_tiny = "microsoft/swin-tiny-patch4-window7-224"

    @classmethod
    def support(cls) -> list[str]:
        return [model.name for model in cls]

    @classmethod
    def check_download(cls):
        from utilities.misc import set_proxy

        set_proxy()
        for model in cls:
            AutoModelForImageClassification.from_pretrained(
                model.value, cache_dir=HF_CACHE_DIR
            )
            print(f"{model} check done")

    def get(
        self, num_classes: int, pretrained: bool = True, base_model: bool = False
    ) -> AutoModelForImageClassification:
        kwargs = {
            **HF_LOAD_KWARGS,
            "pretrained_model_name_or_path": self.value,
            "num_labels": num_classes,
        }

        logging.set_verbosity_error()  # 设置日志级别为仅错误
        
        if pretrained:
            model = AutoModelForImageClassification.from_pretrained(**kwargs)
        else:
            config = AutoConfig.from_pretrained(**kwargs)
            model = AutoModelForImageClassification.from_config(config)

        model.train()
        if base_model:
            model = model.base_model
        return model

    def feat_extractor(self, model: PreTrainedModel) -> "FeatExtractFn":
        """Return the feature extractor function for the model

        Args:
            model (PreTrainedModel): _description_

        Raises:
            ValueError: if the model is not supported

        Returns:
            Callable: A function with signature:
            (x: torch.Tensor, output_heatmap: bool = False, output_featmap) -> (OutputFeatExtractor(feat: torch.Tensor, feat_map: torch.Tensor, heatmaps: torch.Tensor))
        """
        if isinstance(model, ViTPreTrainedModel):
            return partial(vit_feat_extract, model)
        elif isinstance(model, ResNetPreTrainedModel):
            return partial(resnet_feat_extract, model)
        elif isinstance(model, SwinPreTrainedModel):
            return partial(swin_feat_extract, model)
        else:
            raise ValueError(f"Unsupported model: {self.name}")

    @property
    def feat_dim(self):
        feat_map = {
            "resnet18": 512,
            "resnet50": 2048,
            "vit_small": 384,
            "vit_base": 768,
            "swin_tiny": 768,
        }
        return feat_map[self.name]

    @property
    def featmap_size(self):
        featsize_map = {
            "resnet18": 79,
            "resnet50": 79,
            "vit_small": 196,
            "vit_base": 196,
            "swin_tiny": 196,
        }
        return featsize_map[self.name]


def vit_feat_extract(
    model: ViTPreTrainedModel,
    x: torch.Tensor,
    output_heatmap: bool = False,
    output_featmap: bool = False,
):
    output = model(x, output_attentions=True, output_hidden_states=True)
    hidden_states: torch.Tensor = output.hidden_states[-1]
    feat = hidden_states[:, 0]
    feat_dim = hidden_states.shape[-1]
    patch_size = int(hidden_states.shape[1] ** 0.5)

    featmap = (
        (
            hidden_states[:, 1:]
            .reshape(-1, patch_size, patch_size, feat_dim)
            .permute(0, 3, 1, 2)
        )
        if output_featmap
        else None
    )

    heat_maps = (
        output.attentions[-1].mean(1)[:, 0, 1:].reshape(-1, patch_size, patch_size)
        if output_heatmap
        else None
    )
    return OutputFeatExtractor(feat, featmap, heat_maps)


def resnet_feat_extract(
    model: ResNetPreTrainedModel, x: torch.Tensor, output_heatmap: bool = False
):
    output = model(x, output_hidden_states=True)
    feat = torch.flatten(output.pooler_output, 1)
    if not output_heatmap:
        return feat
    heat_maps = output.hidden_states[-1].mean(1)
    return OutputFeatExtractor(feat, output.last_hidden_state, heat_maps)


def swin_feat_extract(
    model: SwinPreTrainedModel, x: torch.Tensor, output_heatmap: bool = False
):
    output = model(x, output_hidden_states=True)
    feat = output.pooler_output
    if not output_heatmap:
        return feat
    featmap = output.reshaped_hidden_states[-1]
    heat_maps = featmap.mean(1)
    return OutputFeatExtractor(feat, featmap, heat_maps)
