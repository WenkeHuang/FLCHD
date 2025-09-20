import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy

import wandb
from backbone.chdbinary import CHDImageDataItem
from backbone.chdmodule.modules.metrics import CHDMetricCLSBinary, CHDMetricCLSMulticlass
from backbone.chdmodule.utilities.enums import Stage
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch
import numpy as np

from utils.constract import DiagColType

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FedProc.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos



class FedProc(FederatedModel):
    NAME = 'fedproc'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedProc, self).__init__(nets_list, args, transform)
        self.global_protos = []
        self.local_protos = {}
        self.chdtype = args.chdtype

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def proto_aggregation(self,local_protos_list):
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


    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        #online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        online_clients = total_clients
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])
        self.global_protos=self.proto_aggregation(self.local_protos)
        self.aggregate_nets(None)
        return None

    def _train_net(self, index, net, train_loader):
        Epoch = self.args.communication_epoch-1
        alpha = 1 - self.epoch_index/Epoch

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

        if self.chdtype == 'binary':
            print("Client using chd type binary")
            metric_train = CHDMetricCLSBinary(stage=f"{Stage.TRAIN}").to(self.device)
        else:
            print("Client using chd type multi-class")
            metric_train = CHDMetricCLSMulticlass(
                DiagColType,
                stage=f"{Stage.TRAIN}"
            ).to(self.device)

        # 将全局原型数据准备好
        all_global_protos_keys = []
        all_f = []
        if len(self.global_protos) != 0:
            all_global_protos_keys = np.array(list(self.global_protos.keys()))
            for protos_key in all_global_protos_keys:
                temp_f = self.global_protos[protos_key]
                # 这里不要转换为 CPU 张量，因为后续会调用 to(self.device)
                temp_f = torch.cat(temp_f, dim=0).to(self.device)
                all_f.append(temp_f)

        iterator = tqdm(range(self.local_epoch))
        for iter in iterator:
            agg_protos_label = {}
            metric_train.reset()
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
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
                lossCE = step_output.loss 
                logits = step_output.model_output['logits'].to(self.device)
                metric_train.update(logits, labels)
                
                f = net.features(images)

                if len(self.global_protos) == 0:
                    loss_InfoNCE = 0 * lossCE
                else:
                    i = 0
                    loss_InfoNCE = None
                    for label in labels:
                        label_item = label.item()
                        if label_item in self.global_protos.keys():
                            # 修复：避免在 NumPy 数组上调用 to() 方法
                            mask = all_global_protos_keys == label_item
                            if np.any(mask):
                                f_pos_idx = np.where(mask)[0][0]
                                f_pos = all_f[f_pos_idx]
                                
                                # 收集所有其他标签的特征
                                negative_indices = np.where(all_global_protos_keys != label_item)[0]
                                f_neg_list = [all_f[idx] for idx in negative_indices]
                                if f_neg_list:  # 确保有负样本
                                    f_neg = torch.cat(f_neg_list)
    
                                    f_now = f[i].unsqueeze(0)
                                    
                                    embedding_len = f_pos.shape
                                    f_neg = f_neg.unsqueeze(1).view(-1, embedding_len[0])
                                    f_pos = f_pos.view(-1, embedding_len[0])
                                    f_proto = torch.cat((f_pos, f_neg), dim=0)
                                    l = torch.cosine_similarity(f_now, f_proto, dim=1)
                                    
                                    exp_l = torch.exp(l)
                                    exp_l = exp_l.view(1, -1)
                                    
                                    pos_mask = [1 for _ in range(f_pos.shape[0])] + [0 for _ in range(f_neg.shape[0])]
                                    pos_mask = torch.tensor(pos_mask, dtype=torch.float).to(self.device)
                                    pos_mask = pos_mask.view(1, -1)
                                    
                                    pos_l = exp_l * pos_mask
                                    sum_pos_l = pos_l.sum(1)
                                    sum_exp_l = exp_l.sum(1)
                                    loss_instance = -torch.log(sum_pos_l / sum_exp_l)
                                    
                                    if loss_InfoNCE is None:
                                        loss_InfoNCE = loss_instance
                                    else:
                                        loss_InfoNCE += loss_instance
                        i += 1
                        
                    # 避免除零错误
                    if loss_InfoNCE is not None and i > 0:
                        loss_InfoNCE = loss_InfoNCE / i
                    else:
                        loss_InfoNCE = 0 * lossCE
    
                # 确保 loss_InfoNCE 是一个张量
                if loss_InfoNCE is None:
                    loss_InfoNCE = 0 * lossCE
    
                loss = alpha * loss_InfoNCE + (1-alpha) * lossCE
                loss.backward()
                iterator.desc = "Local Pariticipant %d CE = %0.3f,InfoNCE = %0.3f" % (index, lossCE, loss_InfoNCE)
                optimizer.step()

                if iter == self.local_epoch-1:
                    for i in range(len(labels)):
                        if labels[i].item() in agg_protos_label:
                            agg_protos_label[labels[i].item()].append(f[i,:])
                        else:
                            agg_protos_label[labels[i].item()] = [f[i,:]]
            # 计算并获取 metrics 结果
            metrics_results = metric_train.compute()
            metric_train.reset()
            print(f"Local Participant {index} Epoch {iter + 1}/{self.local_epoch} metrics results:")
            print('Train/CLS/ACC:', f"{metrics_results['Train/CLS/ACC'].item():.4f}    ",
                  'Train/CLS/AUC:', f"{metrics_results['Train/CLS/AUC'].item():.4f}    ",
                  'Train/CLS/F1:', f"{metrics_results['Train/CLS/F1'].item():.4f}")
            
            # 记录训练指标
            if self.chdtype == 'binary':
                wandb.run.log({
                    f"Client_{index}/ACC": metrics_results["Train/CLS/ACC"].item(),
                    f"Client_{index}/AUC": metrics_results["Train/CLS/AUC"].item(),
                    f"Client_{index}/F1": metrics_results["Train/CLS/F1"].item(),
                    f"Client_{index}/Total_Loss": loss.item(),
                    f"Client_{index}/Step": self.epoch_index * self.local_epoch + iter,
                })
            else:
                log_dict = {}

                for key, value in metrics_results.items():
                    formatted_key = f"Client_{index}/{key}"
                    if hasattr(value, 'item'):
                        log_dict[formatted_key] = value.item()
                    else:
                        log_dict[formatted_key] = value

                additional_metrics = {
                    f"Client_{index}/Total_Loss": loss.item(),
                    f"Client_{index}/Step": self.epoch_index * self.local_epoch + iter,
                }

                log_dict.update(additional_metrics)
                
                wandb.run.log(log_dict)
            
            # 更新调度器（基于损失）
            scheduler.step(loss)
            
        agg_protos = agg_func(agg_protos_label)
        self.local_protos[index] = agg_protos
