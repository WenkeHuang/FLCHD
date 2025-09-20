import os
from collections import namedtuple
from functools import partial

import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import Compose
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from image_transforms import img_eval_aug, img_train_aug
from ..utilities.enums import CLSSetting, CPTSetting, Source, Stage, Task
from ..utilities.logging import get_logger

log = get_logger(__name__)

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


class CHDImageDataModule(LightningDataModule):
    def __init__(
        self,
        cls_setting: str,
        task: str,
        cls_name: list[str],
        cpt_setting: str,
        cpt_name: list[str],
        cpt_pos_weight: list[float],
        cpt_values: dict[str, list[str]],
        col_diag: str,
        apply_cpt_reweight: bool = False,
        drop_hospitals: list[str] | str = None,
        ext_hospitals: list[str] | str = None,
        annotation_file: str = "label/mixup/total_labels-new_qualified.csv",
        img_dir: str = "data/processed/mixup/images",
        mask_dir: str = "data/processed/mixup/masks",
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        img_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 8,
        mixed_img: bool = False,
    ):
        super().__init__()
        self.task = Task(task)
        self.cls_name = cls_name
        self.cpt_name = cpt_name
        self.cpt_values = cpt_values
        self.cpt_pos_weight = cpt_pos_weight
        self.col_diag = col_diag
        self.num_workers = num_workers
        self.int_hospitals: list[str] = None

        self.cls_setting = cls_setting
        self.cpt_setting = cpt_setting
        self.apply_cpt_reweight = apply_cpt_reweight
        self.drop_hospitals = drop_hospitals or []
        self.ext_hospitals = ext_hospitals or []
        if isinstance(self.drop_hospitals, str):
            self.drop_hospitals = [self.drop_hospitals]
        if isinstance(self.ext_hospitals, str):
            self.ext_hospitals = [self.ext_hospitals]
        self.annotation_file = annotation_file
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.img_size = img_size
        self.batch_size = batch_size
        self.mixed_img = mixed_img

        self.annotation = self.load_annotation()
        self.current_source = Source.INTERNAL

    def switch_source(self):
        if Source.EXTERNAL not in self.annotation["Source"].unique():
            raise ValueError("No external data available.")
        self.current_source = Source.EXTERNAL
        log.warning("Switched to external data.")

    def setup(self, stage):
        _dataset = partial(
            CHDImageDataset,
            task=self.task,
            img_dir=self.img_dir,
            mask_dir=self.mask_dir,
            annotation=self.annotation,
            cls_name=self.cls_name,
            cpt_values=self.cpt_values,
            diag_col=self.col_diag,
            mixed_img=self.mixed_img,
            img_size=self.img_size,
        )
        log.info(f">>> Current Source: '{self.current_source}' <<<")
        annotation = self.annotation[self.annotation["Source"] == self.current_source]
        train_trans = img_train_aug(self.img_size)
        eval_trans = img_eval_aug(self.img_size)

        train_df = annotation[annotation["Stage"] == Stage.TRAIN]
        test_df = annotation[annotation["Stage"] == Stage.TEST]
        val_df = annotation[annotation["Stage"] == Stage.VAL]

        if val_df.empty:
            val_df = test_df

        if self.current_source == Source.EXTERNAL and stage != "test":
            raise ValueError(
                f"External data can only be used for testing, but got stage: {stage}"
            )

        if stage == "fit":
            self.trainset: CHDImageDataset = _dataset(
                annotation=train_df, transform=train_trans
            )
            self.valset = _dataset(annotation=val_df, transform=eval_trans)
        elif stage == "validate":
            self.valset = _dataset(annotation=val_df, transform=eval_trans)
        elif stage == "test":
            self.testset = _dataset(annotation=test_df, transform=eval_trans)
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def load_annotation(self):
        annotation = pd.read_csv(self.annotation_file)
        annotation = self.setup_cls_setting(annotation)
        annotation = self.setup_cpt_setting(annotation)
        annotation = self.setup_source(annotation)

        return annotation

    @property
    def label_stats(self):
        stage_order = [Stage.TRAIN, Stage.VAL, Stage.TEST]
        int_label = self.annotation[self.annotation["Source"] == Source.INTERNAL]
        ext_label = self.annotation[self.annotation["Source"] == Source.EXTERNAL]
        cls_stats = label_stats(
            int_label,
            ext_label,
            self.col_diag,
            "Stage",
            self.cls_name,
            stage_order,
        )

        cpt_stats = []
        for cpt in self.cpt_name:
            cpt_stats.append(
                label_stats(
                    int_label,
                    ext_label,
                    cpt,
                    "Stage",
                    row_val_order=stage_order,
                    sum_col_total=False,
                )
            )

        cpt_stats = pd.concat(cpt_stats, axis=1, keys=self.cpt_name)
        cls_stats.columns = pd.MultiIndex.from_product(
            [["Diagnosis"], cls_stats.columns]
        )
        stats = pd.concat([cls_stats, cpt_stats], axis=1).T

        return stats

    def setup_cls_setting(self, annotation: pd.DataFrame):
        if self.cls_setting == CLSSetting.MC3RE:
            self.col_diag = "Multiclass"
            annotation[self.col_diag] = annotation[self.col_diag].apply(
                lambda x: "功能性单心室" if x in ["SV", "房事瓣闭锁"] else x
            )
        elif self.cls_setting == CLSSetting.MC4ORE:
            annotation[self.col_diag] = annotation[self.col_diag].apply(
                lambda x: "功能性单心室" if x in ["SV", "房事瓣闭锁"] else x
            )
            name_cls_wo_other = [_cls for _cls in self.cls_name if _cls != "Other"]
            annotation[self.col_diag] = annotation[self.col_diag].apply(
                lambda x: "Other" if x not in name_cls_wo_other else x
            )

        # filter diag col by name_cls
        annotation = annotation[annotation[self.col_diag].isin(self.cls_name)]
        log.info(f"CLSSetting: {self.cls_setting}, set {self.cls_name=}")

        return annotation

    def setup_cpt_setting(self, annotation: pd.DataFrame):
        annotation["ChamberNum"] = annotation["ChamberNum"].astype(str)
        annotation.loc[:, "ChamberNum"] = annotation["ChamberNum"].apply(
            lambda x: ">=2" if int(x) >= 2 else "<2"
        )

        if self.apply_cpt_reweight:
            log.info(f"Set {self.cpt_pos_weight=}")
        else:
            self.cpt_pos_weight = None
            log.info("No CPT reweighting applied")

        self.cpt_values = {cpt: self.cpt_values[cpt] for cpt in self.cpt_name}
        log.info(f"CPTSetting: {self.cpt_setting}, set {self.cpt_name=}")
        log.info(f"CPT values: {self.cpt_values}")

        return annotation

    def setup_source(self, annotation: pd.DataFrame):
        annotation = annotation[~annotation["Hospital"].isin(self.drop_hospitals)]
        log.info(
            f"Drop hospitals: {self.drop_hospitals}, sample count: {len(annotation)}"
        )

        internal_data = annotation[
            ~annotation["Hospital"].isin(self.ext_hospitals)
        ].copy()
        self.int_hospitals = internal_data["Hospital"].unique().tolist()
        log.info(
            f"Internal hospitals: {self.int_hospitals}, sample count: {len(internal_data)}"
        )

        external_data = annotation[
            annotation["Hospital"].isin(self.ext_hospitals)
        ].copy()
        log.info(
            f"External hospitals: {self.ext_hospitals}, sample count: {len(external_data)}"
        )

        internal_data = self.setup_stage(internal_data)
        external_data["Stage"] = Stage.TEST
        external_data["Source"] = Source.EXTERNAL

        annotation = pd.concat([internal_data, external_data])
        return annotation

    def setup_stage(self, internal_data: pd.DataFrame):
        train_data, test_data = train_test_split(
            internal_data,
            test_size=self.test_ratio,
            stratify=internal_data[self.col_diag],
            random_state=42,
        )
        if self.val_ratio is None:
            log.warning("Using test data as validation data.")
            val_data = test_data
        else:
            train_data, val_data = train_test_split(
                train_data,
                test_size=self.val_ratio / (1 - self.test_ratio),
                stratify=train_data[self.col_diag],
                random_state=42,
            )
        internal_data.loc[train_data.index, "Stage"] = Stage.TRAIN
        internal_data.loc[val_data.index, "Stage"] = Stage.VAL
        internal_data.loc[test_data.index, "Stage"] = Stage.TEST
        internal_data["Source"] = Source.INTERNAL
        return internal_data

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
        )


class CHDImageDataset(Dataset):
    def __init__(
        self,
        task: str,
        img_dir: str,
        mask_dir: str,
        annotation: pd.DataFrame | str,
        cls_name: list[str],
        cpt_values: dict[str, list[str]],
        diag_col: str,
        transform: Compose,
        mixed_img: bool = False,
        img_size: int = 224,
    ):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        if isinstance(annotation, str):
            self.annotation = pd.read_csv(annotation)
        elif isinstance(annotation, pd.DataFrame):
            self.annotation = annotation
        else:
            raise ValueError(
                "annotation_file must be a DataFrame or a path to a CSV file"
            )
        self.cls_name = cls_name
        self.cpt_values = cpt_values
        self.task = task
        self.mixed_img = mixed_img
        self.img_size = img_size
        self.diag_col = diag_col
        self.transform = transform
        self.hospitals = ["BJFC", "GXFY", "CSSFY", "XYYY"]

    def __len__(self) -> int:
        return len(self.annotation)

    def __getitem__(self, idx: int):
        item = self.annotation.iloc[idx]
        item_id = item["ID"]
        target = self.cls_name.index(item[self.diag_col])
        if self.task == Task.BINARY:
            target = torch.FloatTensor([target])

        hospital_id = self.hospitals.index(item["Hospital"])

        concept = []
        for cpt_name in self.cpt_values:
            cpt_value = item[cpt_name]
            cpt_idx = self.cpt_values[cpt_name].index(cpt_value)
            concept.append(cpt_idx)
        concept = torch.FloatTensor(concept)

        img_path = os.path.join(self.img_dir, f"{item_id}.png")
        mask_path = os.path.join(self.mask_dir, f"{item_id}.png")

        bgr_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.mixed_img:
            _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
            img = img * binary_mask[:, :, np.newaxis]

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        return CHDImageDataItem(
            img, concept, mask, target, hospital_id, item_id, img_path, mask_path
        )


def label_stats(
    int_label: pd.DataFrame,
    ext_label: pd.DataFrame,
    target_col: str,
    target_row: str,
    col_val_order: list[str] = None,
    row_val_order: list[str] = None,
    sum_row_total: bool = True,
    sum_col_total: bool = True,
):
    int_label = int_label[[target_row, target_col]]
    int_stats = (
        int_label.value_counts()
        .reset_index()
        .pivot(index=target_row, columns=target_col)
        .droplevel(level=0, axis=1)
    )
    if row_val_order:
        int_stats = int_stats.reindex(row_val_order)
    if col_val_order:
        int_stats = int_stats[col_val_order]
    if sum_col_total:
        int_stats["Total"] = int_stats.sum(axis=1)
    if sum_row_total:
        int_stats.loc["Total"] = int_stats.sum()

    ext_stats = ext_label[target_col].value_counts().to_frame(name="External").T
    if sum_col_total:
        ext_stats["Total"] = ext_stats.sum(axis=1)

    stats = pd.concat([int_stats, ext_stats], axis=0)
    stats = stats.fillna(0).astype(int)

    return stats