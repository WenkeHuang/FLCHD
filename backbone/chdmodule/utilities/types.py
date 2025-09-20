from dataclasses import dataclass

import torch


@dataclass
class StepOutput:
    loss: torch.Tensor
    loss_dict: dict[str, torch.Tensor] = None
    model_output: dict[str, torch.Tensor] = None

    def __post_init__(self):
        if self.loss_dict is None:
            self.loss_dict = {}
        if self.model_output is None:
            self.model_output = {}

    def to_dict(self):
        return {"loss": self.loss, **self.model_output, **self.loss_dict}
