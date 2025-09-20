import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
import numpy as np

import wandb
from backbone.chdbinary import CHDImageDataItem
from backbone.chdmodule.modules.metrics import CHDMetricCLSBinary, CHDMetricCLSMulticlass
from backbone.chdmodule.utilities.enums import Stage
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch

from utils.constract import DiagColType

# https://github.com/yuetan031/fedproto

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FedProto.')
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


class FedProto(FederatedModel):
    NAME = 'fedproto'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedProto, self).__init__(nets_list, args, transform)
        self.mu = args.mu
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
                loss_mse = nn.MSELoss()
                if len(self.global_protos) == 0:
                    lossProto = 0*lossCE
                else:
                    f_new = copy.deepcopy(f.data)
                    i = 0
                    for label in labels:
                        if label.item() in self.global_protos.keys():
                            f_new[i, :] = self.global_protos[label.item()][0].data
                        i += 1
                    lossProto = loss_mse(f_new, f)

                lossProto = lossProto * self.mu

                loss = lossCE + lossProto
                loss.backward()
                iterator.desc = "Local Pariticipant %d CE = %0.3f,Proto = %0.3f" % (index, lossCE, lossProto)
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
