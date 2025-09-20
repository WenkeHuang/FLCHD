import torchvision.transforms as transforms
from backbone.chdbinary import FLCPTBinary
from backbone.chdproto import FLCPTProto
from backbone.chdraw import FLCPTRaw
from utils.conf import data_path
from datasets.utils.federated_dataset import FederatedDataset, partition_office_domain_skew_loaders_new
from datasets.transforms.denormalization import DeNormalize
from backbone.ResNet import resnet10, resnet12, resnet18, resnet34
from backbone.efficientnet import EfficientNetB0
from backbone.googlenet import GoogLeNet
from backbone.mobilnet_v2 import MobileNetV2
from backbone.chd import FLCPT
from torchvision.datasets import ImageFolder, DatasetFolder

import albumentations as A 
from albumentations.pytorch import ToTensorV2 
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

from chdtools.image_transforms import img_eval_aug, img_train_aug
from chdtools.enums import CLSSetting, CPTSetting, Source, Stage, Task, Hospital
from chdtools.logging import get_logger

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

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
img_size = 224
TaskKind = Task.MULTICLASS
ClsSetting = CLSSetting.MC6
cpt_setting = CPTSetting.ALLCPT

DiagCol = "MC5"
ClsName = ["Normal", "FSV", "AVSD", "HV"]

MainDatasetsDir = "/data0/lyx_mydata/chd_mixup_split"
TestPercent = 0.2
CptValues = {
            "ChamberNum": ["<2", ">=2"],
            "FlowNum": ["Single", "Two"],
            "FlowPattern": ["Unequal", "Equal"],
            }
TrainCompose = A.Compose(
        [
            A.ToGray(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussianBlur(p=0.2),
            A.RandomResizedCrop(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ]
    )
TestCompose = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ]
    )
apply_cpt_reweight = False
cpt_name = ["ChamberNum", "FlowNum", "FlowPattern"]
cpt_pos_weight = [0.006, 0.23, 0.23, 0.018]
drop_hospitals = []
ext_hospitals = []

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
        annotation_file: str = "label/mixup/fl-label-fix.csv",
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

        self.current_source = Source.INTERNAL

    def switch_source(self):
        if Source.EXTERNAL not in self.annotation["Source"].unique():
            raise ValueError("No external data available.")
        self.current_source = Source.EXTERNAL

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
        annotation = annotation[annotation["Hospital"].isin(["BJFC", "CSSFY", "GXFY", "XYYY"])]
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
            self.col_diag = "MC6"
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

        annotation = annotation[annotation[self.col_diag].isin(self.cls_name)]

        return annotation

    def setup_cpt_setting(self, annotation: pd.DataFrame):
        annotation["ChamberNum"] = annotation["ChamberNum"].astype(str)
        annotation.loc[:, "ChamberNum"] = annotation["ChamberNum"].apply(
            lambda x: ">=2" if int(x) >= 2 else "<2"
        )

        if not self.apply_cpt_reweight:
            self.cpt_pos_weight = None

        self.cpt_values = {cpt: self.cpt_values[cpt] for cpt in self.cpt_name}

        return annotation

    def setup_source(self, annotation: pd.DataFrame):
        annotation = annotation[~annotation["Hospital"].isin(self.drop_hospitals)]

        internal_data = annotation[
            ~annotation["Hospital"].isin(self.ext_hospitals)
        ].copy()
        self.int_hospitals = internal_data["Hospital"].unique().tolist()

        external_data = annotation[
            annotation["Hospital"].isin(self.ext_hospitals)
        ].copy()

        internal_data = self.setup_stage(internal_data)
        external_data["Stage"] = Stage.TEST
        external_data["Source"] = Source.EXTERNAL

        annotation = pd.concat([internal_data, external_data])
        return annotation

    def setup_stage(self, internal_data: pd.DataFrame):
        case_ids = internal_data['CaseID'].unique()
        id_to_diag = {}
        for case_id in case_ids:
            diag = internal_data[internal_data['CaseID'] == case_id][self.col_diag].iloc[0]
            id_to_diag[case_id] = diag

        id_diag_series = pd.Series(id_to_diag)

        try:
            train_case_ids, test_case_ids = train_test_split(
                case_ids,
                test_size=self.test_ratio,
                stratify=id_diag_series,
                random_state=42
            )
        except ValueError:
            train_case_ids, test_case_ids = train_test_split(
                case_ids,
                test_size=self.test_ratio,
                random_state=42
            )

        train_indices = internal_data[internal_data['CaseID'].isin(train_case_ids)].index
        test_indices = internal_data[internal_data['CaseID'].isin(test_case_ids)].index

        if self.val_ratio is None:
            val_indices = test_indices
        else:
            train_id_diag = {case_id: id_to_diag[case_id] for case_id in train_case_ids}
            train_id_diag_series = pd.Series(train_id_diag)

            try:
                final_train_case_ids, val_case_ids = train_test_split(
                    train_case_ids,
                    test_size=self.val_ratio / (1 - self.test_ratio),
                    stratify=train_id_diag_series,
                    random_state=42
                )
            except ValueError:
                final_train_case_ids, val_case_ids = train_test_split(
                    train_case_ids,
                    test_size=self.val_ratio / (1 - self.test_ratio),
                    random_state=42
                )


        # 验证没有CaseID出现在多个集合中
        if self.val_ratio is not None:
            train_cases = set(internal_data.loc[train_indices, 'CaseID'])
            val_cases = set(internal_data.loc[val_indices, 'CaseID'])
            test_cases = set(internal_data.loc[test_indices, 'CaseID'])

            train_val_overlap = train_cases.intersection(val_cases)
            train_test_overlap = train_cases.intersection(test_cases)
            val_test_overlap = val_cases.intersection(test_cases)

        return internal_data

    def train_dataloader(self) -> DataLoader:
        indices = np.arange(len(self.trainset))
        train_sampler = torch.utils.data.SubsetRandomSampler(indices)

        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=train_sampler,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        indices = np.arange(len(self.valset))
        val_sampler = torch.utils.data.SubsetRandomSampler(indices)

        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=val_sampler,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        indices = np.arange(len(self.testset))
        test_sampler = torch.utils.data.SubsetRandomSampler(indices)

        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=test_sampler,
            pin_memory=False,
            drop_last=False,
        )

class FedCHDNew(FederatedDataset):
    NAME = 'fl_chd_multi'
    SETTING = 'domain_skew'
    DOMAINS_LIST = ['BJFC', 'CSSFY', 'GXFY', 'XYYY']
    percent_dict = {'BJFC': 0.25, 'GXFY': 0.25, 'CSSFY': 0.25, 'XYYY': 0.25}
    
    N_SAMPLES_PER_Class = None
    N_CLASS = 6

    Nor_TRANSFORM = TrainCompose
    
    def __init__(self, args):
        super().__init__(args)
        self.cls_name = ClsName
        self.cpt_values = CptValues
        self.diag_col = DiagCol
        self.annotation_file = "label/mixup/fl-label-fix.csv"
        self.task = TaskKind 
        self.cls_setting = ClsSetting
        self.col_diag = DiagCol
        self.apply_cpt_reweight = apply_cpt_reweight
        self.cpt_name = cpt_name
        self.cpt_setting = cpt_setting
        self.cpt_pos_weight = cpt_pos_weight
        self.drop_hospitals = drop_hospitals
        self.ext_hospitals = ext_hospitals
        
        self.leave_one_out = getattr(args, 'leave_one_out', 'false') == 'true'
        self.leave_out_hospital = getattr(args, 'leave_out_hospital', 'CSSFY')
        
        if self.leave_one_out:
            self.client_domains = [h for h in self.DOMAINS_LIST if h != self.leave_out_hospital]
            print(f"Leave-one-out mode enabled. Using {self.client_domains} as clients, {self.leave_out_hospital} as test hospital.")
        else:
            self.client_domains = self.DOMAINS_LIST
            print(f"Standard mode enabled. Using all hospitals {self.client_domains} as clients.")
        
        self.client_data_modules = {}
        self.leave_out_data_module = None
        
        for hospital in self.client_domains:
            hospital_annotation = pd.read_csv(self.annotation_file)
            hospital_annotation = hospital_annotation[hospital_annotation["Hospital"] == hospital]
            
            self.client_data_modules[hospital] = CHDImageDataModule(
                cls_setting=self.cls_setting,
                task=self.task,
                cls_name=self.cls_name,
                cpt_setting=self.cpt_setting,
                cpt_name=self.cpt_name,
                cpt_pos_weight=self.cpt_pos_weight,
                cpt_values=self.cpt_values,
                col_diag=self.col_diag,
                apply_cpt_reweight=False,
                img_dir=f"{MainDatasetsDir}/{hospital}/images",
                mask_dir=f"{MainDatasetsDir}/{hospital}/masks",
                test_ratio=0.2,
                val_ratio=0.1,
                img_size=224,
                batch_size=32,
                num_workers=8
            )

            annotation = self.client_data_modules[hospital].setup_cls_setting(hospital_annotation)
            annotation = self.client_data_modules[hospital].setup_cpt_setting(annotation)
            annotation = self.client_data_modules[hospital].setup_source(annotation)
            self.client_data_modules[hospital].annotation = annotation

            self.client_data_modules[hospital].setup("fit")
            self.client_data_modules[hospital].setup("test")
            
            print(f"Initialized data module for client {hospital} with {len(hospital_annotation)} samples")
        
        if self.leave_one_out:
            leave_out_annotation = pd.read_csv(self.annotation_file)
            leave_out_annotation = leave_out_annotation[leave_out_annotation["Hospital"] == self.leave_out_hospital]
            
            self.leave_out_data_module = CHDImageDataModule(
                cls_setting=self.cls_setting,
                task=self.task,
                cls_name=self.cls_name,
                cpt_setting=self.cpt_setting,
                cpt_name=self.cpt_name,
                cpt_pos_weight=self.cpt_pos_weight,
                cpt_values=self.cpt_values,
                col_diag=self.col_diag,
                apply_cpt_reweight=False,
                img_dir=f"{MainDatasetsDir}/{self.leave_out_hospital}/images",
                mask_dir=f"{MainDatasetsDir}/{self.leave_out_hospital}/masks",
                test_ratio=0.2,
                val_ratio=0.1,
                img_size=224,
                batch_size=32,
                num_workers=8
            )
            
            annotation = self.leave_out_data_module.setup_cls_setting(leave_out_annotation)
            annotation = self.leave_out_data_module.setup_cpt_setting(annotation)
            annotation = self.leave_out_data_module.setup_source(annotation)
            self.leave_out_data_module.annotation = annotation
            
            self.leave_out_data_module.setup("test")
            
            print(f"Initialized leave-one-out test data module for {self.leave_out_hospital} with {len(leave_out_annotation)} samples")
        
    def get_data_loaders(self, selected_domain_list=None):
        if self.leave_one_out:
            if selected_domain_list is None or len(selected_domain_list) == 0:
                using_list = self.client_domains
            else:
                using_list = [domain for domain in selected_domain_list if domain != self.leave_out_hospital]
        else:
            if selected_domain_list is None or len(selected_domain_list) == 0:
                using_list = self.DOMAINS_LIST
            else:
                using_list = selected_domain_list
        
        train_loaders = []
        test_loaders = []
        
        for domain in using_list:
            if domain in self.client_data_modules:
                train_loader = self.client_data_modules[domain].train_dataloader()
                train_loaders.append(train_loader)
                
                test_loader = self.client_data_modules[domain].test_dataloader()
                test_loaders.append(test_loader)
        
        return train_loaders, test_loaders

    def get_leave_one_out_test_loader(self):
        if self.leave_one_out and self.leave_out_data_module is not None:
            return self.leave_out_data_module.test_dataloader()
        else:
            raise ValueError("Leave-one-out mode is not enabled or leave-out data module is not initialized")

    @staticmethod
    def get_transform():
        transform = transforms.Compose([
            transforms.ToPILImage(),
            FedCHDNew.Nor_TRANSFORM
        ])
        return transform

    @staticmethod
    def get_backbone(parti_num, names_list=None):
        nets_dict = {
            'resnet10': resnet10,
            'resnet12': resnet12,
            'resnet18': resnet18,
            'resnet34': resnet34,
            'efficient': EfficientNetB0,
            'mobilnet': MobileNetV2,
            'googlenet': GoogLeNet,
            "cpt": FLCPTProto,
            "cptbinary": FLCPTRaw
        }
  
        nets_list = []
        if names_list is None:
            print(f"Initialize with default model FLCPTRaw, based with vit_base for {parti_num} clients.", "cls_name:", len(ClsName), "cpt_name:", len(cpt_name))
            for j in range(parti_num):
                nets_list.append(FLCPTRaw(
                    base_model_name="vit_base",
                    cls_name=ClsName,
                    cpt_name=cpt_name,
                )) 
        else:
            for j in range(parti_num):
                net_name = names_list[j]
                nets_list.append(nets_dict[net_name](FedCHDNew.N_CLASS))
        return nets_list

    @staticmethod
    def get_normalization_transform():
        print("get_normalization_transform")
        transform = transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD
        )
        return transform

    @staticmethod
    def get_denormalization_transform():
        print("get_denormalization_transform")
        transform = DeNormalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD
        )
        return transform