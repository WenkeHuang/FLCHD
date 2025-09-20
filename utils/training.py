import os
import torch
import time
from argparse import Namespace
import numpy as np
from collections import Counter
import torch.nn.functional as F

import wandb
from backbone.chdmodule.modules.metrics import CHDMetricCLSBinary, CHDMetricCLSMulticlass, CHDMetricCPTMultilabel, CHDClinicalMetrics
from backbone.chdmodule.utilities.enums import Stage
from models.utils.federated_model import FederatedModel
from datasets.utils.federated_dataset import FederatedDataset
from typing import Tuple, Dict, List
from torch.utils.data import DataLoader
from utils.constract import DiagColType
from utils.logger import CsvWriter
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from utils.a_test_validation import ATestValidation, FederatedATestValidation, ATestResults

now_time = time.strftime("%Y%m%d_%H%M%S")

def global_evaluate(model: FederatedModel, test_dl: DataLoader, setting: str, name: str, args: Namespace) -> Tuple[list, list]:
    accs = []
    metrics_results = {}
    net = model.global_net
    status = net.training
    net.eval()

    if args.chdtype == 'binary':
        print("Global using chd type binary")
        metric_test = CHDMetricCLSBinary(stage=f"{Stage.TEST}").to(model.device)
        clinical_metric = None 
    else:
        print("Global using chd type multi-class")
        metric_test = CHDMetricCLSMulticlass(
            DiagColType,
            stage=f"{Stage.TEST}"
        ).to(model.device)
        clinical_metric = CHDClinicalMetrics(
            DiagColType,
            stage=f"{Stage.TEST}"
        ).to(model.device)
    
    cpt_names = ["ChamberNum", "FlowNum", "FlowPattern"]

    metric_cpt = CHDMetricCPTMultilabel(
        cpt_names,
        stage=f"{Stage.TEST}"
    ).to(model.device)
    
    all_labels = []
    all_probs = []

    for j, dl in enumerate(test_dl):
        correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
        class_counts = Counter()
        
        for batch_idx, batch in enumerate(dl):
            with torch.no_grad():
                image, labels = batch.image.to(model.device), batch.target.to(model.device)
                concepts = batch.concept.to(model.device)
                
                class_counts.update(labels.cpu().numpy())

                output = net(image)
                outputs = output.logits.to(model.device)
                metric_test.update(outputs, labels)
                
                if clinical_metric is not None:
                    clinical_metric.update(outputs, labels)
                
                if args.chdtype != 'binary': 
                    probs = F.softmax(outputs, dim=1)
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())
                

                num_classes = outputs.size(1)
                k = min(5, num_classes)
                _, max_k = torch.topk(outputs, k, dim=-1)
                labels = labels.view(-1, 1)
                top1 += (labels == max_k[:, 0:1]).sum().item()
                
                if k > 1:
                    top5 += (labels == max_k).sum().item()
                else:
                    top5 += top1
                    
                total += labels.size(0)
                
        top1acc = round(100 * top1 / total, 2)
        top5acc = round(100 * top5 / total, 2)
        
        accs.append(top1acc)
        
        print(f"\nTest Loader {j} Class Distribution:")
        for class_idx, count in sorted(class_counts.items()):
            class_name = DiagColType(class_idx).name if hasattr(DiagColType, 'name') else f"Class_{class_idx}"
            print(f"{class_name}: {count} samples")
        
    metrics_results = metric_test.compute()
    metric_test.reset()
    
    if clinical_metric is not None:
        clinical_results = clinical_metric.compute()
        print(f"Debug: Clinical metrics computed: {list(clinical_results.keys())}")
        metrics_results.update(clinical_results)
        clinical_metric.reset()
    
    if args.chdtype != 'binary' and all_labels and all_probs:
        all_probs = np.vstack(all_probs)
        target_classes = ['AVSD', 'FSV', 'HV']
        generate_roc_curves(np.array(all_labels), all_probs, args.output_dir, model.epoch_index, target_classes)
    
    cpt_metrics = metric_cpt.compute()
    for key, value in cpt_metrics.items():
        metrics_results[f"Test/CPT/{key}"] = value.item()
           
    metric_cpt.reset()
    net.train(status)
    return accs, metrics_results


def train(model: FederatedModel, private_dataset: FederatedDataset,
          args: Namespace) -> None:
    if args.csv_log:
        csv_writer = CsvWriter(args, private_dataset)

    model.N_CLASS = private_dataset.N_CLASS
    domains_list = private_dataset.DOMAINS_LIST
    domains_len = len(domains_list)

    print("train begin, model class", model.N_CLASS, "domains_list", domains_list)
    
    if args.rand_dataset:
        max_num = 10
        is_ok = False

        while not is_ok:
            if model.args.dataset == 'fl_officecaltech':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num - domains_len, replace=True, p=None)
                selected_domain_list = list(selected_domain_list) + domains_list
            elif model.args.dataset == 'fl_digits':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num, replace=True, p=None)
            else:
                print('using force equally distributed domains, as default setting.')
                selected_domain_list = domains_list

            result = dict(Counter(selected_domain_list))

            for k in result:
                if result[k] > max_num:
                    is_ok = False
                    break
            else:
                is_ok = True

    else:
        selected_domain_dict = {'mnist': 6, 'usps': 4, 'svhn': 3, 'syn': 7} 
        
        selected_domain_list = []
        for k in selected_domain_dict:
            domain_num = selected_domain_dict[k]
            for i in range(domain_num):
                selected_domain_list.append(k)

        selected_domain_list = np.random.permutation(selected_domain_list)

        result = Counter(selected_domain_list)
    
    print('Selected Domain Count:', result, 'With List Descripted:', selected_domain_list)

    pri_train_loaders, test_loaders = private_dataset.get_data_loaders(selected_domain_list)
    model.trainloaders = pri_train_loaders
    if hasattr(model, 'ini'):
        model.ini()

    accs_dict = {}
    mean_accs_list = []

   
    model_save_dir = os.path.join(args.output_dir, 'saved_models', args.model, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(model_save_dir, exist_ok=True)
    
    
    best_acc = 0.0
    best_auc = 0.0
    best_f1 = 0.0
    best_epoch = 0

    atest_results_history = []

    Epoch = args.communication_epoch
    for epoch_index in range(Epoch):
        model.epoch_index = epoch_index
        if hasattr(model, 'loc_update'):
            epoch_loc_loss_dict = model.loc_update(pri_train_loaders)

        accs, metrics = global_evaluate(model, test_loaders, private_dataset.SETTING, private_dataset.NAME, args)
        mean_acc = round(np.mean(accs, axis=0), 3)

        if run_atest and (epoch_index % atest_interval == 0 or epoch_index == Epoch - 1):
            print(f"\n[Epoch {epoch_index}] Running A-Test Validation...")
            atest_results = run_atest_validation(
                model=model,
                test_loaders=test_loaders,
                args=args,
                epoch=epoch_index,
                save_results=True
            )
            atest_results_history.append({
                'epoch': epoch_index,
                'results': atest_results
            })

            metrics['Test/ATest/IR'] = torch.tensor(atest_results['mean_ir'])
            metrics['Test/ATest/SR'] = torch.tensor(atest_results['mean_sr'])
            metrics['Test/ATest/LC'] = torch.tensor(atest_results['learning_capacity'])
            metrics['Test/ATest/CI_lower'] = torch.tensor(atest_results['confidence_interval'][0])
            metrics['Test/ATest/CI_upper'] = torch.tensor(atest_results['confidence_interval'][1])
        mean_accs_list.append(mean_acc)
        for i in range(len(accs)):
            if i in accs_dict:
                accs_dict[i].append(accs[i])
            else:
                accs_dict[i] = [accs[i]]

        print('The ' + str(epoch_index) + ' Communcation Accuracy:', str(mean_acc), 'Method:', model.args.model)
        print(accs)
        print('Test/CLS/ACC:', f"{metrics['Test/CLS/ACC'].item():.4f}    ",
                'Test/CLS/AUC:', f"{metrics['Test/CLS/AUC'].item():.4f}    ",
                'Test/CLS/F1:', f"{metrics['Test/CLS/F1'].item():.4f}    ")
        print('Test/CLS/SENS:', f"{metrics['Test/CLS/SENS'].item():.4f}   ",
                'Test/CLS/SPEC:', f"{metrics['Test/CLS/SPEC'].item():.4f}   ",
                'Test/CLS/PPV:', f"{metrics.get('Test/CLS/PPV', torch.tensor(0.0)).item():.4f}")
        
        if args.chdtype != 'binary' and any(key.startswith('Test/Clinical/') for key in metrics.keys()):
            print("\n--- CHD Clinical Screening Metrics ---")
            print('Severe CHD Sensitivity:', f"{metrics.get('Test/Clinical/SevereCHD_SENS', torch.tensor(0.0)).item():.4f}   ",
                  'Severe CHD Specificity:', f"{metrics.get('Test/Clinical/SevereCHD_SPEC', torch.tensor(0.0)).item():.4f}")
            print('Severe CHD PPV:', f"{metrics.get('Test/Clinical/SevereCHD_PPV', torch.tensor(0.0)).item():.4f}   ",
                  'Normal NPV:', f"{metrics.get('Test/Clinical/Normal_NPV', torch.tensor(0.0)).item():.4f}")
            print('Severe CHD Miss Rate:', f"{metrics.get('Test/Clinical/SevereCHD_MissRate', torch.tensor(0.0)).item():.4f}   ",
                  'Clinical Risk Score:', f"{metrics.get('Test/Clinical/RiskScore', torch.tensor(0.0)).item():.4f}")
            
            print("\n--- Per-Class Balanced Accuracy ---")
            for cls_name in ["Normal", "FSV", "AVSD", "HV"]:
                bal_acc_key = f"Test/Clinical/{cls_name}_BalACC"
                if bal_acc_key in metrics:
                    print(f'{cls_name} Balanced ACC:', f"{metrics[bal_acc_key].item():.4f}")
            print()
        
        
        current_acc = metrics['Test/CLS/ACC'].item()
        current_auc = metrics['Test/CLS/AUC'].item()
        current_f1 = metrics['Test/CLS/F1'].item()
        
        is_best = False
        if current_f1 > best_f1:
            best_acc = current_acc
            best_auc = current_auc
            best_f1 = current_f1
            best_epoch = epoch_index
            is_best = True