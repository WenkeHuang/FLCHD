import torch
from torchmetrics import (
    AUROC,
    Accuracy,
    F1Score,
    Metric,
    MetricCollection,
    Recall,
    Specificity,
    Precision,
    MatthewsCorrCoef,
)
import torch.nn.functional as F
import numpy as np


class CHDMetricCLSBinary(Metric):
    def __init__(self, stage: str = None):
        super().__init__()
        self.metric = MetricCollection(
            {
                "ACC": Accuracy(task="binary"),
                "F1": F1Score(task="binary"),
                "SENS": Recall(task="binary"),
                "SPEC": Specificity(task="binary"),
                "PPV": Precision(task="binary"),
                "AUC": AUROC(task="binary"),
            },
            prefix=f"{stage}/CLS/",
        )

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.metric.update(pred, target.to(int))

    def compute(self):
        return self.metric.compute()

    def reset(self):
        self.metric.reset()


class CHDClinicalMetrics(Metric):
    """
    Specialized metrics for CHD screening focusing on clinical risk assessment
    """
    def __init__(self, name_cls: list[str], stage: str = None):
        super().__init__()
        self.name_cls = name_cls
        self.num_cls = len(name_cls)
        self.stage = stage
        
        # Define which classes are considered severe CHD (non-normal)
        self.normal_idx = None
        self.severe_chd_indices = []
        
        for i, cls_name in enumerate(name_cls):
            if cls_name.lower() in ['normal', 'norm']:
                self.normal_idx = i
            else:
                self.severe_chd_indices.append(i)
        
        # Add state for computing custom metrics
        self.add_state("predictions", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # Store predictions and targets for custom metric computation
        if len(preds.shape) > 1 and preds.shape[1] > 1:
            # Convert logits to predicted classes
            pred_classes = torch.argmax(preds, dim=1)
        else:
            pred_classes = preds
            
        self.predictions.append(pred_classes.cpu())
        self.targets.append(targets.cpu())
    
    def compute(self):
        all_preds = torch.cat(self.predictions)
        all_targets = torch.cat(self.targets)
        
        results = {}
        
        # 1. Severe CHD Sensitivity (Recall for non-normal classes)
        severe_chd_sensitivity = self._compute_severe_chd_sensitivity(all_preds, all_targets)
        results[f"{self.stage}/Clinical/SevereCHD_SENS"] = severe_chd_sensitivity
        
        # 2. Severe CHD Specificity (correctly identifying normal as normal)
        severe_chd_specificity = self._compute_severe_chd_specificity(all_preds, all_targets)
        results[f"{self.stage}/Clinical/SevereCHD_SPEC"] = severe_chd_specificity
        
        # 3. Severe CHD PPV (Precision for severe CHD detection)
        severe_chd_ppv = self._compute_severe_chd_ppv(all_preds, all_targets)
        results[f"{self.stage}/Clinical/SevereCHD_PPV"] = severe_chd_ppv
        
        # 4. Normal NPV (Negative Predictive Value for normal cases)
        normal_npv = self._compute_normal_npv(all_preds, all_targets)
        results[f"{self.stage}/Clinical/Normal_NPV"] = normal_npv
        
        # 5. Balanced Accuracy for each class
        balanced_accs = self._compute_balanced_accuracy_per_class(all_preds, all_targets)
        for i, cls_name in enumerate(self.name_cls):
            results[f"{self.stage}/Clinical/{cls_name}_BalACC"] = balanced_accs[i]
        
        # 6. Clinical Risk Score (weighted by severity)
        clinical_risk_score = self._compute_clinical_risk_score(all_preds, all_targets)
        results[f"{self.stage}/Clinical/RiskScore"] = clinical_risk_score
        
        # 7. Miss Rate for severe CHD (1 - sensitivity for severe CHD)
        miss_rate = 1.0 - severe_chd_sensitivity
        results[f"{self.stage}/Clinical/SevereCHD_MissRate"] = miss_rate
        
        return results
    
    def _compute_severe_chd_sensitivity(self, preds, targets):
        """Compute sensitivity for detecting any severe CHD (non-normal classes)"""
        if not self.severe_chd_indices:
            return torch.tensor(0.0)
        
        # True positives: correctly identified severe CHD cases
        severe_chd_mask = torch.zeros_like(targets, dtype=torch.bool)
        for idx in self.severe_chd_indices:
            severe_chd_mask |= (targets == idx)
        
        if severe_chd_mask.sum() == 0:
            return torch.tensor(0.0)
        
        # Predicted as any severe CHD type
        severe_pred_mask = torch.zeros_like(preds, dtype=torch.bool)
        for idx in self.severe_chd_indices:
            severe_pred_mask |= (preds == idx)
        
        true_positives = (severe_chd_mask & severe_pred_mask).sum().float()
        actual_positives = severe_chd_mask.sum().float()
        
        return true_positives / actual_positives if actual_positives > 0 else torch.tensor(0.0)
    
    def _compute_severe_chd_specificity(self, preds, targets):
        """Compute specificity for normal cases (correctly identifying normal as normal)"""
        if self.normal_idx is None:
            return torch.tensor(0.0)
        
        normal_mask = (targets == self.normal_idx)
        if normal_mask.sum() == 0:
            return torch.tensor(0.0)
        
        normal_pred_mask = (preds == self.normal_idx)
        true_negatives = (normal_mask & normal_pred_mask).sum().float()
        actual_negatives = normal_mask.sum().float()
        
        return true_negatives / actual_negatives if actual_negatives > 0 else torch.tensor(0.0)
    
    def _compute_severe_chd_ppv(self, preds, targets):
        """Compute PPV for severe CHD detection"""
        if not self.severe_chd_indices:
            return torch.tensor(0.0)
        
        # Predicted as any severe CHD
        severe_pred_mask = torch.zeros_like(preds, dtype=torch.bool)
        for idx in self.severe_chd_indices:
            severe_pred_mask |= (preds == idx)
        
        if severe_pred_mask.sum() == 0:
            return torch.tensor(0.0)
        
        # Actually severe CHD
        severe_true_mask = torch.zeros_like(targets, dtype=torch.bool)
        for idx in self.severe_chd_indices:
            severe_true_mask |= (targets == idx)
        
        true_positives = (severe_pred_mask & severe_true_mask).sum().float()
        predicted_positives = severe_pred_mask.sum().float()
        
        return true_positives / predicted_positives if predicted_positives > 0 else torch.tensor(0.0)
    
    def _compute_normal_npv(self, preds, targets):
        """Compute NPV for normal cases"""
        if self.normal_idx is None:
            return torch.tensor(0.0)
        
        normal_pred_mask = (preds == self.normal_idx)
        if normal_pred_mask.sum() == 0:
            return torch.tensor(0.0)
        
        normal_true_mask = (targets == self.normal_idx)
        true_negatives = (normal_pred_mask & normal_true_mask).sum().float()
        predicted_negatives = normal_pred_mask.sum().float()
        
        return true_negatives / predicted_negatives if predicted_negatives > 0 else torch.tensor(0.0)
    
    def _compute_balanced_accuracy_per_class(self, preds, targets):
        """Compute balanced accuracy for each class"""
        balanced_accs = []
        
        for class_idx in range(self.num_cls):
            class_mask = (targets == class_idx)
            if class_mask.sum() == 0:
                balanced_accs.append(torch.tensor(0.0))
                continue
            
            # True positives and false negatives for this class
            tp = ((preds == class_idx) & (targets == class_idx)).sum().float()
            fn = ((preds != class_idx) & (targets == class_idx)).sum().float()
            
            # True negatives and false positives for this class
            tn = ((preds != class_idx) & (targets != class_idx)).sum().float()
            fp = ((preds == class_idx) & (targets != class_idx)).sum().float()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else torch.tensor(0.0)
            
            balanced_acc = (sensitivity + specificity) / 2.0
            balanced_accs.append(balanced_acc)
        
        return torch.stack(balanced_accs)
    
    def _compute_clinical_risk_score(self, preds, targets):
        """
        Compute a clinical risk score that heavily weights missing severe CHD cases
        """
        if not self.severe_chd_indices or self.normal_idx is None:
            return torch.tensor(0.0)
        
        # Define risk weights (higher weight = higher clinical risk if missed)
        risk_weights = torch.ones(self.num_cls)
        
        # Weight severe CHD classes higher based on clinical severity
        for i, cls_name in enumerate(self.name_cls):
            if i == self.normal_idx:
                risk_weights[i] = 0.1  # Low penalty for misclassifying normal
            elif any(severe_type in cls_name.upper() for severe_type in ['FSV', 'SV']):
                risk_weights[i] = 5.0  # Very high penalty for missing single ventricle
            elif 'AVSD' in cls_name.upper():
                risk_weights[i] = 3.0  # High penalty for missing AVSD
            elif 'HV' in cls_name.upper():
                risk_weights[i] = 4.0  # Very high penalty for missing hypoplastic ventricles
            else:
                risk_weights[i] = 2.0  # Moderate penalty for other severe CHD
        
        # Compute weighted accuracy
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for class_idx in range(self.num_cls):
            class_mask = (targets == class_idx)
            if class_mask.sum() == 0:
                continue
            
            correct = ((preds == class_idx) & (targets == class_idx)).sum().float()
            total = class_mask.sum().float()
            weight = risk_weights[class_idx]
            
            class_score = (correct / total) * weight
            total_weighted_score += class_score
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else torch.tensor(0.0)
    
    def reset(self):
        self.predictions.clear()
        self.targets.clear()


class CHDMetricCLSMulticlass(Metric):
    def __init__(self, name_cls: list[str], stage: str = None):
        super().__init__()
        self.name_cls = name_cls
        num_cls = len(name_cls)
        self.metric = MetricCollection(
            {
                "NonAvg/ACC": Accuracy(
                    task="multiclass", num_classes=num_cls, average="none"
                ),
                "NonAvg/F1": F1Score(
                    task="multiclass", num_classes=num_cls, average="none"
                ),
                "NonAvg/SENS": Recall(
                    task="multiclass", num_classes=num_cls, average="none"
                ),
                "NonAvg/SPEC": Specificity(
                    task="multiclass", num_classes=num_cls, average="none"
                ),
                "NonAvg/PPV": Precision(
                    task="multiclass", num_classes=num_cls, average="none"
                ),
                "NonAvg/AUC": AUROC(
                    task="multiclass", num_classes=num_cls, average="none"
                ),
                "CLS/ACC": Accuracy(
                    task="multiclass", num_classes=num_cls, average="micro"
                ),
                "CLS/F1": F1Score(
                    task="multiclass", num_classes=num_cls, average="macro"
                ),
                "CLS/SENS": Recall(
                    task="multiclass", num_classes=num_cls, average="macro"
                ),
                "CLS/SPEC": Specificity(
                    task="multiclass", num_classes=num_cls, average="macro"
                ),
                "CLS/PPV": Precision(
                    task="multiclass", num_classes=num_cls, average="macro"
                ),
                "CLS/AUC": AUROC(
                    task="multiclass", num_classes=num_cls, average="macro"
                ),
            },
            prefix=f"{stage}/",
        )

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.metric.update(pred, target.to(int))

    def compute(self):
        ret = self.metric.compute()
        avg_val, non_avg_val = {}, {}
        for key, value in ret.items():
            if "NonAvg" in key:
                non_avg_val[key] = value
            else:
                avg_val[key] = value
        for key, value in non_avg_val.items():
            for idx, v in enumerate(value):
                avg_val[key.replace("NonAvg", self.name_cls[idx])] = v
        return avg_val

    def reset(self):
        self.metric.reset()


class CHDClinicalMetrics(Metric):
    """
    Specialized metrics for CHD screening focusing on clinical risk assessment
    """
    def __init__(self, name_cls: list[str], stage: str = None):
        super().__init__()
        self.name_cls = name_cls
        self.num_cls = len(name_cls)
        self.stage = stage
        
        # Define which classes are considered severe CHD (non-normal)
        self.normal_idx = None
        self.severe_chd_indices = []
        
        for i, cls_name in enumerate(name_cls):
            if cls_name.lower() in ['normal', 'norm']:
                self.normal_idx = i
            else:
                self.severe_chd_indices.append(i)
        
        # Add state for computing custom metrics
        self.add_state("predictions", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # Store predictions and targets for custom metric computation
        if len(preds.shape) > 1 and preds.shape[1] > 1:
            # Convert logits to predicted classes
            pred_classes = torch.argmax(preds, dim=1)
        else:
            pred_classes = preds
            
        self.predictions.append(pred_classes.cpu())
        self.targets.append(targets.cpu())
    
    def compute(self):
        all_preds = torch.cat(self.predictions)
        all_targets = torch.cat(self.targets)
        
        results = {}
        
        # 1. Severe CHD Sensitivity (Recall for non-normal classes)
        severe_chd_sensitivity = self._compute_severe_chd_sensitivity(all_preds, all_targets)
        results[f"{self.stage}/Clinical/SevereCHD_SENS"] = severe_chd_sensitivity
        
        # 2. Severe CHD Specificity (correctly identifying normal as normal)
        severe_chd_specificity = self._compute_severe_chd_specificity(all_preds, all_targets)
        results[f"{self.stage}/Clinical/SevereCHD_SPEC"] = severe_chd_specificity
        
        # 3. Severe CHD PPV (Precision for severe CHD detection)
        severe_chd_ppv = self._compute_severe_chd_ppv(all_preds, all_targets)
        results[f"{self.stage}/Clinical/SevereCHD_PPV"] = severe_chd_ppv
        
        # 4. Normal NPV (Negative Predictive Value for normal cases)
        normal_npv = self._compute_normal_npv(all_preds, all_targets)
        results[f"{self.stage}/Clinical/Normal_NPV"] = normal_npv
        
        # 5. Balanced Accuracy for each class
        balanced_accs = self._compute_balanced_accuracy_per_class(all_preds, all_targets)
        for i, cls_name in enumerate(self.name_cls):
            results[f"{self.stage}/Clinical/{cls_name}_BalACC"] = balanced_accs[i]
        
        # 6. Clinical Risk Score (weighted by severity)
        clinical_risk_score = self._compute_clinical_risk_score(all_preds, all_targets)
        results[f"{self.stage}/Clinical/RiskScore"] = clinical_risk_score
        
        # 7. Miss Rate for severe CHD (1 - sensitivity for severe CHD)
        miss_rate = 1.0 - severe_chd_sensitivity
        results[f"{self.stage}/Clinical/SevereCHD_MissRate"] = miss_rate
        
        return results
    
    def _compute_severe_chd_sensitivity(self, preds, targets):
        """Compute sensitivity for detecting any severe CHD (non-normal classes)"""
        if not self.severe_chd_indices:
            return torch.tensor(0.0)
        
        # True positives: correctly identified severe CHD cases
        severe_chd_mask = torch.zeros_like(targets, dtype=torch.bool)
        for idx in self.severe_chd_indices:
            severe_chd_mask |= (targets == idx)
        
        if severe_chd_mask.sum() == 0:
            return torch.tensor(0.0)
        
        # Predicted as any severe CHD type
        severe_pred_mask = torch.zeros_like(preds, dtype=torch.bool)
        for idx in self.severe_chd_indices:
            severe_pred_mask |= (preds == idx)
        
        true_positives = (severe_chd_mask & severe_pred_mask).sum().float()
        actual_positives = severe_chd_mask.sum().float()
        
        return true_positives / actual_positives if actual_positives > 0 else torch.tensor(0.0)
    
    def _compute_severe_chd_specificity(self, preds, targets):
        """Compute specificity for normal cases (correctly identifying normal as normal)"""
        if self.normal_idx is None:
            return torch.tensor(0.0)
        
        normal_mask = (targets == self.normal_idx)
        if normal_mask.sum() == 0:
            return torch.tensor(0.0)
        
        normal_pred_mask = (preds == self.normal_idx)
        true_negatives = (normal_mask & normal_pred_mask).sum().float()
        actual_negatives = normal_mask.sum().float()
        
        return true_negatives / actual_negatives if actual_negatives > 0 else torch.tensor(0.0)
    
    def _compute_severe_chd_ppv(self, preds, targets):
        """Compute PPV for severe CHD detection"""
        if not self.severe_chd_indices:
            return torch.tensor(0.0)
        
        # Predicted as any severe CHD
        severe_pred_mask = torch.zeros_like(preds, dtype=torch.bool)
        for idx in self.severe_chd_indices:
            severe_pred_mask |= (preds == idx)
        
        if severe_pred_mask.sum() == 0:
            return torch.tensor(0.0)
        
        # Actually severe CHD
        severe_true_mask = torch.zeros_like(targets, dtype=torch.bool)
        for idx in self.severe_chd_indices:
            severe_true_mask |= (targets == idx)
        
        true_positives = (severe_pred_mask & severe_true_mask).sum().float()
        predicted_positives = severe_pred_mask.sum().float()
        
        return true_positives / predicted_positives if predicted_positives > 0 else torch.tensor(0.0)
    
    def _compute_normal_npv(self, preds, targets):
        """Compute NPV for normal cases"""
        if self.normal_idx is None:
            return torch.tensor(0.0)
        
        normal_pred_mask = (preds == self.normal_idx)
        if normal_pred_mask.sum() == 0:
            return torch.tensor(0.0)
        
        normal_true_mask = (targets == self.normal_idx)
        true_negatives = (normal_pred_mask & normal_true_mask).sum().float()
        predicted_negatives = normal_pred_mask.sum().float()
        
        return true_negatives / predicted_negatives if predicted_negatives > 0 else torch.tensor(0.0)
    
    def _compute_balanced_accuracy_per_class(self, preds, targets):
        """Compute balanced accuracy for each class"""
        balanced_accs = []
        
        for class_idx in range(self.num_cls):
            class_mask = (targets == class_idx)
            if class_mask.sum() == 0:
                balanced_accs.append(torch.tensor(0.0))
                continue
            
            # True positives and false negatives for this class
            tp = ((preds == class_idx) & (targets == class_idx)).sum().float()
            fn = ((preds != class_idx) & (targets == class_idx)).sum().float()
            
            # True negatives and false positives for this class
            tn = ((preds != class_idx) & (targets != class_idx)).sum().float()
            fp = ((preds == class_idx) & (targets != class_idx)).sum().float()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else torch.tensor(0.0)
            
            balanced_acc = (sensitivity + specificity) / 2.0
            balanced_accs.append(balanced_acc)
        
        return torch.stack(balanced_accs)
    
    def _compute_clinical_risk_score(self, preds, targets):
        """
        Compute a clinical risk score that heavily weights missing severe CHD cases
        """
        if not self.severe_chd_indices or self.normal_idx is None:
            return torch.tensor(0.0)
        
        # Define risk weights (higher weight = higher clinical risk if missed)
        risk_weights = torch.ones(self.num_cls)
        
        # Weight severe CHD classes higher based on clinical severity
        for i, cls_name in enumerate(self.name_cls):
            if i == self.normal_idx:
                risk_weights[i] = 0.1  # Low penalty for misclassifying normal
            elif any(severe_type in cls_name.upper() for severe_type in ['FSV', 'SV']):
                risk_weights[i] = 5.0  # Very high penalty for missing single ventricle
            elif 'AVSD' in cls_name.upper():
                risk_weights[i] = 3.0  # High penalty for missing AVSD
            elif 'HV' in cls_name.upper():
                risk_weights[i] = 4.0  # Very high penalty for missing hypoplastic ventricles
            else:
                risk_weights[i] = 2.0  # Moderate penalty for other severe CHD
        
        # Compute weighted accuracy
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for class_idx in range(self.num_cls):
            class_mask = (targets == class_idx)
            if class_mask.sum() == 0:
                continue
            
            correct = ((preds == class_idx) & (targets == class_idx)).sum().float()
            total = class_mask.sum().float()
            weight = risk_weights[class_idx]
            
            class_score = (correct / total) * weight
            total_weighted_score += class_score
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else torch.tensor(0.0)
    
    def reset(self):
        self.predictions.clear()
        self.targets.clear()


class CHDMetricCPTMultilabel(Metric):
    def __init__(self, name_cpt: list[str], stage: str = None):
        super().__init__()
        self.name_cpt = name_cpt
        num_labels = len(name_cpt)
        self.metric = MetricCollection(
            {
                #"NonAvg/ACC": Accuracy(
                #    task="multilabel", num_labels=num_labels, average="none"
                #),
                "NonAvg/F1": F1Score(
                    task="multilabel", num_labels=num_labels, average="none"
                ),
                #"NonAvg/SENS": Recall(
                #    task="multilabel", num_labels=num_labels, average="none"
                #),
                #"NonAvg/SPEC": Specificity(
                #    task="multilabel", num_labels=num_labels, average="none"
                #),
                #"NonAvg/AUC": AUROC(
                #    task="multilabel", num_labels=num_labels, average="none"
                #),
                #"CPTMacro/ACC": Accuracy(
                #    task="multilabel", num_labels=num_labels, average="macro"
                #),
                "CPTMacro/F1": F1Score(
                    task="multilabel", num_labels=num_labels, average="macro"
                ),
                #"CPTMacro/SENS": Recall(
                #    task="multilabel", num_labels=num_labels, average="macro"
                #),
                #"CPTMacro/SPEC": Specificity(
                #    task="multilabel", num_labels=num_labels, average="macro"
                #),
                #"CPTMacro/AUC": AUROC(
                #    task="multilabel", num_labels=num_labels, average="macro"
                #),
                #"CPTMicro/ACC": Accuracy(
                #    task="multilabel", num_labels=num_labels, average="micro"
                #),
                "CPTMicro/F1": F1Score(
                    task="multilabel", num_labels=num_labels, average="micro"
                ),
                #"CPTMicro/SENS": Recall(
                #    task="multilabel", num_labels=num_labels, average="micro"
                #),
                #"CPTMicro/SPEC": Specificity(
                #    task="multilabel", num_labels=num_labels, average="micro"
                #),
            },
            prefix=f"{stage}/",
        )

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.metric.update(pred, target.to(int))

    def compute(self):
        ret = self.metric.compute()
        avg_val, non_avg_val = {}, {}
        for key, value in ret.items():
            if "NonAvg" in key:
                non_avg_val[key] = value
            else:
                avg_val[key] = value
        for key, value in non_avg_val.items():
            for idx, v in enumerate(value):
                avg_val[key.replace("NonAvg", self.name_cpt[idx])] = v
        return avg_val

    def reset(self):
        self.metric.reset()


class CHDClinicalMetrics(Metric):
    """
    Specialized metrics for CHD screening focusing on clinical risk assessment
    """
    def __init__(self, name_cls: list[str], stage: str = None):
        super().__init__()
        self.name_cls = name_cls
        self.num_cls = len(name_cls)
        self.stage = stage
        
        # Define which classes are considered severe CHD (non-normal)
        self.normal_idx = None
        self.severe_chd_indices = []
        
        for i, cls_name in enumerate(name_cls):
            if cls_name.lower() in ['normal', 'norm']:
                self.normal_idx = i
            else:
                self.severe_chd_indices.append(i)
        
        # Add state for computing custom metrics
        self.add_state("predictions", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # Store predictions and targets for custom metric computation
        if len(preds.shape) > 1 and preds.shape[1] > 1:
            # Convert logits to predicted classes
            pred_classes = torch.argmax(preds, dim=1)
        else:
            pred_classes = preds
            
        self.predictions.append(pred_classes.cpu())
        self.targets.append(targets.cpu())
    
    def compute(self):
        all_preds = torch.cat(self.predictions)
        all_targets = torch.cat(self.targets)
        
        results = {}
        
        # 1. Severe CHD Sensitivity (Recall for non-normal classes)
        severe_chd_sensitivity = self._compute_severe_chd_sensitivity(all_preds, all_targets)
        results[f"{self.stage}/Clinical/SevereCHD_SENS"] = severe_chd_sensitivity
        
        # 2. Severe CHD Specificity (correctly identifying normal as normal)
        severe_chd_specificity = self._compute_severe_chd_specificity(all_preds, all_targets)
        results[f"{self.stage}/Clinical/SevereCHD_SPEC"] = severe_chd_specificity
        
        # 3. Severe CHD PPV (Precision for severe CHD detection)
        severe_chd_ppv = self._compute_severe_chd_ppv(all_preds, all_targets)
        results[f"{self.stage}/Clinical/SevereCHD_PPV"] = severe_chd_ppv
        
        # 4. Normal NPV (Negative Predictive Value for normal cases)
        normal_npv = self._compute_normal_npv(all_preds, all_targets)
        results[f"{self.stage}/Clinical/Normal_NPV"] = normal_npv
        
        # 5. Balanced Accuracy for each class
        balanced_accs = self._compute_balanced_accuracy_per_class(all_preds, all_targets)
        for i, cls_name in enumerate(self.name_cls):
            results[f"{self.stage}/Clinical/{cls_name}_BalACC"] = balanced_accs[i]
        
        # 6. Clinical Risk Score (weighted by severity)
        clinical_risk_score = self._compute_clinical_risk_score(all_preds, all_targets)
        results[f"{self.stage}/Clinical/RiskScore"] = clinical_risk_score
        
        # 7. Miss Rate for severe CHD (1 - sensitivity for severe CHD)
        miss_rate = 1.0 - severe_chd_sensitivity
        results[f"{self.stage}/Clinical/SevereCHD_MissRate"] = miss_rate
        
        return results
    
    def _compute_severe_chd_sensitivity(self, preds, targets):
        """Compute sensitivity for detecting any severe CHD (non-normal classes)"""
        if not self.severe_chd_indices:
            return torch.tensor(0.0)
        
        # True positives: correctly identified severe CHD cases
        severe_chd_mask = torch.zeros_like(targets, dtype=torch.bool)
        for idx in self.severe_chd_indices:
            severe_chd_mask |= (targets == idx)
        
        if severe_chd_mask.sum() == 0:
            return torch.tensor(0.0)
        
        # Predicted as any severe CHD type
        severe_pred_mask = torch.zeros_like(preds, dtype=torch.bool)
        for idx in self.severe_chd_indices:
            severe_pred_mask |= (preds == idx)
        
        true_positives = (severe_chd_mask & severe_pred_mask).sum().float()
        actual_positives = severe_chd_mask.sum().float()
        
        return true_positives / actual_positives if actual_positives > 0 else torch.tensor(0.0)
    
    def _compute_severe_chd_specificity(self, preds, targets):
        """Compute specificity for normal cases (correctly identifying normal as normal)"""
        if self.normal_idx is None:
            return torch.tensor(0.0)
        
        normal_mask = (targets == self.normal_idx)
        if normal_mask.sum() == 0:
            return torch.tensor(0.0)
        
        normal_pred_mask = (preds == self.normal_idx)
        true_negatives = (normal_mask & normal_pred_mask).sum().float()
        actual_negatives = normal_mask.sum().float()
        
        return true_negatives / actual_negatives if actual_negatives > 0 else torch.tensor(0.0)
    
    def _compute_severe_chd_ppv(self, preds, targets):
        """Compute PPV for severe CHD detection"""
        if not self.severe_chd_indices:
            return torch.tensor(0.0)
        
        # Predicted as any severe CHD
        severe_pred_mask = torch.zeros_like(preds, dtype=torch.bool)
        for idx in self.severe_chd_indices:
            severe_pred_mask |= (preds == idx)
        
        if severe_pred_mask.sum() == 0:
            return torch.tensor(0.0)
        
        # Actually severe CHD
        severe_true_mask = torch.zeros_like(targets, dtype=torch.bool)
        for idx in self.severe_chd_indices:
            severe_true_mask |= (targets == idx)
        
        true_positives = (severe_pred_mask & severe_true_mask).sum().float()
        predicted_positives = severe_pred_mask.sum().float()
        
        return true_positives / predicted_positives if predicted_positives > 0 else torch.tensor(0.0)
    
    def _compute_normal_npv(self, preds, targets):
        """Compute NPV for normal cases"""
        if self.normal_idx is None:
            return torch.tensor(0.0)
        
        normal_pred_mask = (preds == self.normal_idx)
        if normal_pred_mask.sum() == 0:
            return torch.tensor(0.0)
        
        normal_true_mask = (targets == self.normal_idx)
        true_negatives = (normal_pred_mask & normal_true_mask).sum().float()
        predicted_negatives = normal_pred_mask.sum().float()
        
        return true_negatives / predicted_negatives if predicted_negatives > 0 else torch.tensor(0.0)
    
    def _compute_balanced_accuracy_per_class(self, preds, targets):
        """Compute balanced accuracy for each class"""
        balanced_accs = []
        
        for class_idx in range(self.num_cls):
            class_mask = (targets == class_idx)
            if class_mask.sum() == 0:
                balanced_accs.append(torch.tensor(0.0))
                continue
            
            # True positives and false negatives for this class
            tp = ((preds == class_idx) & (targets == class_idx)).sum().float()
            fn = ((preds != class_idx) & (targets == class_idx)).sum().float()
            
            # True negatives and false positives for this class
            tn = ((preds != class_idx) & (targets != class_idx)).sum().float()
            fp = ((preds == class_idx) & (targets != class_idx)).sum().float()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else torch.tensor(0.0)
            
            balanced_acc = (sensitivity + specificity) / 2.0
            balanced_accs.append(balanced_acc)
        
        return torch.stack(balanced_accs)
    
    def _compute_clinical_risk_score(self, preds, targets):
        """
        Compute a clinical risk score that heavily weights missing severe CHD cases
        """
        if not self.severe_chd_indices or self.normal_idx is None:
            return torch.tensor(0.0)
        
        # Define risk weights (higher weight = higher clinical risk if missed)
        risk_weights = torch.ones(self.num_cls)
        
        # Weight severe CHD classes higher based on clinical severity
        for i, cls_name in enumerate(self.name_cls):
            if i == self.normal_idx:
                risk_weights[i] = 0.1  # Low penalty for misclassifying normal
            elif any(severe_type in cls_name.upper() for severe_type in ['FSV', 'SV']):
                risk_weights[i] = 5.0  # Very high penalty for missing single ventricle
            elif 'AVSD' in cls_name.upper():
                risk_weights[i] = 3.0  # High penalty for missing AVSD
            elif 'HV' in cls_name.upper():
                risk_weights[i] = 4.0  # Very high penalty for missing hypoplastic ventricles
            else:
                risk_weights[i] = 2.0  # Moderate penalty for other severe CHD
        
        # Compute weighted accuracy
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for class_idx in range(self.num_cls):
            class_mask = (targets == class_idx)
            if class_mask.sum() == 0:
                continue
            
            correct = ((preds == class_idx) & (targets == class_idx)).sum().float()
            total = class_mask.sum().float()
            weight = risk_weights[class_idx]
            
            class_score = (correct / total) * weight
            total_weighted_score += class_score
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else torch.tensor(0.0)
    
    def reset(self):
        self.predictions.clear()
        self.targets.clear()
