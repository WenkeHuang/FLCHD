from lightning import Callback

from ..backbone.chdmodule.data.image_data import CHDImageDataItem, CHDImageDataModule
from ..backbone.chdmodule.modules.metrics import (
    CHDMetricCLSBinary,
    CHDMetricCLSMulticlass,
    CHDMetricCPTMultilabel,
)
from ..backbone.chdmodule.utilities.enums import Stage, Task


class CLSMetricCaculator(Callback):
    def __init__(self):
        super().__init__()

    def setup(self, trainer, pl_module, stage):
        datamodule: CHDImageDataModule = trainer.datamodule
        self.current_source = datamodule.current_source
        self.name_cls = datamodule.cls_name
        self.task = datamodule.task

    def on_fit_start(self, trainer, pl_module):
        if self.task == Task.BINARY:
            self.metric_train = CHDMetricCLSBinary(stage=Stage.TRAIN).to(
                pl_module.device
            )
            self.metric_val = CHDMetricCLSBinary(stage=Stage.VAL).to(pl_module.device)
        elif self.task == Task.MULTICLASS:
            self.metric_train = CHDMetricCLSMulticlass(
                self.name_cls, stage=Stage.TRAIN
            ).to(pl_module.device)
            self.metric_val = CHDMetricCLSMulticlass(self.name_cls, stage=Stage.VAL).to(
                pl_module.device
            )

    def on_test_start(self, trainer, pl_module):
        stage = f"{self.current_source}/{Stage.TEST}"
        if self.task == Task.BINARY:
            self.metric_test = CHDMetricCLSBinary(stage=stage).to(pl_module.device)
        elif self.task == Task.MULTICLASS:
            self.metric_test = CHDMetricCLSMulticlass(self.name_cls, stage=stage).to(
                pl_module.device
            )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch: CHDImageDataItem, batch_idx
    ):
        self.metric_train.update(outputs["logits"], batch.target)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch: CHDImageDataItem, batch_idx
    ):
        self.metric_val.update(outputs["logits"], batch.target)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch: CHDImageDataItem, batch_idx
    ):
        self.metric_test.update(outputs["logits"], batch.target)

    def on_train_epoch_end(self, trainer, pl_module):
        train_metrics = self.metric_train.compute()
        val_metrics = self.metric_val.compute()
        self.metric_train.reset()
        self.metric_val.reset()

        pl_module.log_dict(train_metrics, sync_dist=True)
        pl_module.log_dict(val_metrics, sync_dist=True)

    def on_test_epoch_end(self, trainer, pl_module):
        test_metrics = self.metric_test.compute()
        self.metric_test.reset()

        pl_module.log_dict(test_metrics, sync_dist=True)


class CPTMultilabelMetricCalculator(Callback):
    def __init__(self):
        super().__init__()

    def setup(self, trainer, pl_module, stage):
        datamodule: CHDImageDataModule = trainer.datamodule
        self.current_source = datamodule.current_source
        self.cpt_name = datamodule.cpt_name

    def on_fit_start(self, trainer, pl_module):
        self.metric_train = CHDMetricCPTMultilabel(self.cpt_name, stage=Stage.TRAIN).to(
            pl_module.device
        )
        self.metric_val = CHDMetricCPTMultilabel(self.cpt_name, stage=Stage.VAL).to(
            pl_module.device
        )

    def on_test_start(self, trainer, pl_module):
        stage = f"{self.current_source}_{Stage.TEST}"
        self.metric_test = CHDMetricCPTMultilabel(self.cpt_name, stage=stage).to(
            pl_module.device
        )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch: CHDImageDataItem, batch_idx
    ):
        self.metric_train.update(outputs["logits_cpt"], batch.concept)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch: CHDImageDataItem, batch_idx
    ):
        self.metric_val.update(outputs["logits_cpt"], batch.concept)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch: CHDImageDataItem, batch_idx
    ):
        self.metric_test.update(outputs["logits_cpt"], batch.concept)

    def on_train_epoch_end(self, trainer, pl_module):
        train_metrics = self.metric_train.compute()
        val_metrics = self.metric_val.compute()
        self.metric_train.reset()
        self.metric_val.reset()

        pl_module.log_dict(train_metrics, sync_dist=True)
        pl_module.log_dict(val_metrics, sync_dist=True)

    def on_test_epoch_end(self, trainer, pl_module):
        test_metrics = self.metric_test.compute()
        self.metric_test.reset()

        pl_module.log_dict(test_metrics, sync_dist=True)
