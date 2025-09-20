import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from sklearn.decomposition import PCA

import wandb
from backbone.chdmodule.modules.metrics import CHDMetricCLSBinary, CHDMetricCLSMulticlass, CHDMetricCPTMultilabel
from backbone.chdmodule.utilities.enums import Stage
from datasets.chd import CHDImageDataItem
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch

from utils.constract import DiagColType

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='FedCHD')
    add_management_args(parser)
    add_experiment_args(parser)
    
    parser.add_argument('--temperature', type=float, default=1, 
                        help='Temperature parameter for contrastive loss')
    
    parser.add_argument('--vis_dir', type=str, default='visualization_results', 
                        help='Directory to save visualization results')
    parser.add_argument('--vis_freq', type=int, default=1, 
                        help='Frequency of visualization (every N epochs)')
    return parser

def agg_func(protos):
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos

def WireframeSphere(center=[0, 0, 0], radius=1, n_meridians=20, n_circles_latitude=None):
    if n_circles_latitude is None:
        n_circles_latitude = max(n_meridians//2, 4)
        
    u, v = np.mgrid[0:2*np.pi:n_meridians*1j, 0:np.pi:n_circles_latitude*1j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    
    return x, y, z


class FedProtoChdMemoAll(FederatedModel):
    NAME = 'fedprotochdmemoall'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedProtoChdMemoAll, self).__init__(nets_list, args, transform)
        self.mu = args.mu
        self.global_protos = {} 
        self.local_protos = {}
        self.chdtype = args.chdtype
        self.momentum = args.momentum
        self.global_concept_protos = {'pos': None, 'neg': None} 
        self.temperature = args.temperature 
        
        self.vis_dir = args.vis_dir if hasattr(args, 'vis_dir') else 'visualization_results'
        self.vis_freq = args.vis_freq if hasattr(args, 'vis_freq') else 1
        
        self.accumulated_features = {} 
        self.class_names = ["Normal", "Functional Single Ventricle", "Atrioventricular Defect", "Ventricular Disease"]
        self.class_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'] 
        
        os.makedirs(self.vis_dir, exist_ok=True)
        
        import matplotlib.font_manager as fm
        
        plt.rcParams['axes.unicode_minus'] = False

        self.concept_proto_ema_factor = 0.9
        self.concept_proto_loss_history = []
        self.contrastive_ema_factor = 0.9  
        self.contrastive_loss_history = [] 
        
        self.current_epoch_proto_distribution = {}

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def proto_aggregation(self, local_protos_list):
        agg_protos_label = dict()
        for idx in self.online_clients:
            local_protos = local_protos_list[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                agg_protos_label[label] = [proto / len(proto_list)]
            else:
                agg_protos_label[label] = [proto_list[0].data]

        return agg_protos_label

    def concept_proto_aggregation(self):
        pos_protos_sum = None
        neg_protos_sum = None
        client_count = 0

        for idx in self.online_clients:
            client_count += 1
            net = self.nets_list[idx].to(self.device)
            
            proto_predictor = net.model.proto_predictor
            pos_protos = proto_predictor.pos_prototypes.data.clone().detach()  
            neg_protos = proto_predictor.neg_prototypes.data.clone().detach()  
            
            if pos_protos_sum is None:
                pos_protos_sum = pos_protos.clone()
                neg_protos_sum = neg_protos.clone()
            else:
                pos_protos_sum += pos_protos
                neg_protos_sum += neg_protos
        
        if client_count > 0:
            pos_protos_avg = pos_protos_sum / client_count
            neg_protos_avg = neg_protos_sum / client_count
            
            if self.global_concept_protos['pos'] is not None:
                momentum = self.concept_proto_ema_factor 
                self.global_concept_protos['pos'] = momentum * self.global_concept_protos['pos'].clone().detach() + (1 - momentum) * pos_protos_avg.clone().detach()
                self.global_concept_protos['neg'] = momentum * self.global_concept_protos['neg'].clone().detach() + (1 - momentum) * neg_protos_avg.clone().detach()
            else:
                self.global_concept_protos['pos'] = pos_protos_avg.clone().detach()
                self.global_concept_protos['neg'] = neg_protos_avg.clone().detach()
            
            for idx in range(len(self.nets_list)):
                net = self.nets_list[idx].to(self.device)
                with torch.no_grad():
                    net.model.proto_predictor.pos_prototypes.data.copy_(self.global_concept_protos['pos'])
                    net.model.proto_predictor.neg_prototypes.data.copy_(self.global_concept_protos['neg'])

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = total_clients
        self.online_clients = online_clients

        print('Using forced sorted online_clients enqueue:', online_clients)

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])
        
        self.global_protos = self.proto_aggregation(self.local_protos)
        self.concept_proto_aggregation()
        self.aggregate_nets(None)
        return None

    def loss(self, features, labels, global_protos):
        
        features = F.normalize(features, p=2, dim=1)
        
        proto_features = []
        proto_labels = []
        
        
        if targets.sum() == 0:
            return torch.tensor(0.0, device=features.device)
            
        exp_logits = torch.exp(logits)
        pos_exp_logits = torch.sum(exp_logits * targets.float(), dim=1)
        
        neg_exp_logits = torch.sum(exp_logits * (~targets).float(), dim=1)
        
        denominator = pos_exp_logits + neg_exp_logits
        
        eps = 1e-7
        loss = -torch.log(pos_exp_logits / (denominator + eps) + eps)
        
        return loss.mean()

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        optimizer = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',          
            factor=0.5,          
            patience=5,          
            threshold=1e-4,      
            threshold_mode='abs',
            cooldown=0,          
            min_lr=1e-6          
        )
    
        momentum = self.momentum
        contrastive_velocity = None

        if self.chdtype == 'binary':
            print("Client using chd type binary")
            metric_train = CHDMetricCLSBinary(stage=f"{Stage.TRAIN}").to(self.device)
        else:
            print("Client using chd type multi-class")
            metric_train = CHDMetricCLSMulticlass(
                DiagColType,
                stage=f"{Stage.TRAIN}"
            ).to(self.device)
        
        cpt_names = ["ChamberNum", "FlowNum", "FlowPattern"]
        
        metric_cpt = CHDMetricCPTMultilabel(
            cpt_names,
            stage=f"{Stage.TRAIN}"
        ).to(self.device)
    
        iterator = tqdm(range(self.local_epoch))
    
        for iter in iterator:
            agg_protos_label = {}
            metric_train.reset()
    
            for batch_idx, batch in enumerate(train_loader):
                images = batch.image.to(self.device)
                labels = batch.target.to(self.device)
                concepts = batch.concept.to(self.device)
                masks = batch.mask.to(self.device)
    
                gpu_batch = CHDImageDataItem(
                    image=images,
                    concept=concepts,
                    mask=masks,
                    target=labels,
                    hospital_id=batch.hospital_id, 
                    id=batch.id,                   
                    img_path=batch.img_path,       
                    mask_path=batch.mask_path      
                )

                    
                step_output = net.model.shared_step(gpu_batch, batch_idx)
                loss = step_output.loss 
                lossCE = step_output.loss_dict['Concept']
                lossProto = step_output.loss_dict['CLS']
                lossGuide = step_output.loss_dict['Guide']
    
                logits = step_output.model_output['logits'].to(self.device)
                metric_train.update(logits, labels)
                
                logits_cpt = step_output.model_output['logits_cpt'].to(self.device)   
                metric_cpt.update(logits_cpt, concepts)
    
                f = net.features(images)
                
        
            print(f"Local Participant {index} Epoch {iter + 1}/{self.local_epoch} metrics results:")
            print('Train/CLS/ACC:', f"{metrics_results['Train/CLS/ACC'].item():.4f}",
                  'Train/CLS/AUC:', f"{metrics_results['Train/CLS/AUC'].item():.4f}",
                  'Train/CLS/F1:', f"{metrics_results['Train/CLS/F1'].item():.4f}")

            if self.chdtype == 'binary':
                wandb.run.log({
                    f"Client_{index}/ACC": metrics_results["Train/CLS/ACC"].item(),
                    f"Client_{index}/AUC": metrics_results["Train/CLS/AUC"].item(),
                    f"Client_{index}/F1": metrics_results["Train/CLS/F1"].item(),
                    f"Client_{index}/CE_Loss": lossCE,
                    f"Client_{index}/Proto_Loss": lossProto,
                    f"Client_{index}/Guide_Loss": lossGuide,
                    f"Client_{index}/Contrastive_Loss": lossContrastive_value,
                    f"Client_{index}/Contrastive_Raw": contrastive_velocity if contrastive_velocity is not None else 0.0,
                    f"Client_{index}/ConceptProto_Loss": lossConceptProto_value,
                    f"Client_{index}/Total_Loss": loss.item(),
                    f"Client_{index}/Step": self.epoch_index * self.local_epoch + iter,
                    f"Client_{index}/Momentum": momentum
                })
            else:
                log_dict = {}

                for key, value in metrics_results.items():
                    formatted_key = f"Client_{index}/{key}"
                    if hasattr(value, 'item'):
                        log_dict[formatted_key] = value.item()
                    else:
                        log_dict[formatted_key] = value
                        
                for key, value in cpt_metrics_results.items():
                    formatted_key = f"Client_{index}/CPT/{key}"
                    if hasattr(value, 'item'):
                        log_dict[formatted_key] = value.item()
                    else:
                        log_dict[formatted_key] = value
                
                additional_metrics = {
                    f"Client_{index}/CE_Loss": lossCE.item(),
                    f"Client_{index}/Contrastive_Loss": lossContrastive_value,
                    f"Client_{index}/Contrastive_Raw": contrastive_velocity if contrastive_velocity is not None else 0.0,
                    f"Client_{index}/ConceptProto_Loss": lossConceptProto_value,
                    f"Client_{index}/Total_Loss": loss.item(),
                    f"Client_{index}/CE_Loss": lossCE.item(),
                    f"Client_{index}/Proto_Loss": lossProto.item(),
                    f"Client_{index}/Guide_Loss": lossGuide.item(),
                    f"Client_{index}/Step": self.epoch_index * self.local_epoch + iter,
                    f"Client_{index}/Momentum": momentum
                }

                log_dict.update(additional_metrics)
                
                wandb.run.log(log_dict)
            
            scheduler.step(loss)
    
        agg_protos = agg_func(agg_protos_label)
        self.local_protos[index] = agg_protos