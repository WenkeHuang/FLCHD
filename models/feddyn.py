import torch
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
from utils.constract import DiagColType


# https://github.com/katsura-jp/fedavg.pytorch
# https://github.com/vaseline555/Federated-Averaging-PyTorch
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via Fedavg.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class FedDyn(FederatedModel):
    NAME = 'feddyn'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedDyn, self).__init__(nets_list, args, transform)
        self.client_grads = {}
        self.chdtype = args.chdtype

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

        for i in range(len(self.nets_list)):
            self.client_grads[i] = self.build_grad_dict(self.global_net)

    def build_grad_dict(self, model):
        grad_dict = {}
        for key, params in model.state_dict().items():
            grad_dict[key] = torch.zeros_like(params)
        return grad_dict

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        # online_clients = self.random_state.choice(total_clients,self.online_num,replace=False).tolist()
        online_clients = total_clients
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])
        self.aggregate_nets(None)
        return None

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
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

        local_grad = copy.deepcopy(self.client_grads[index])

        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        for epoch in iterator:
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

                reg_loss = 0.0
                cnt = 0.0
                for name, param in self.global_net.named_parameters():
                    term1 = (param * (
                            local_grad[name] - self.global_net.state_dict()[name]
                    )).sum()
                    term2 = (param * param).sum()

                    reg_loss += self.args.reg_lamb * (term1 + term2)
                    cnt += 1.0

                loss=loss+reg_loss/cnt

                logits = step_output.model_output['logits'].to(self.device)
                metric_train.update(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                
            # 计算并获取 metrics 结果
            metrics_results = metric_train.compute()
            metric_train.reset()
            print(f"Local Participant {index} Epoch {epoch + 1}/{self.local_epoch} metrics results:")
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
                    f"Client_{index}/Step": self.epoch_index * self.local_epoch + epoch,
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
                    f"Client_{index}/Step": self.epoch_index * self.local_epoch + epoch,
                }

                log_dict.update(additional_metrics)
                
                wandb.run.log(log_dict)
        
            # 更新调度器（基于损失）
            scheduler.step(loss)

        for name, param in net.named_parameters():
            local_grad[name] += (
                    net.state_dict()[name] - self.global_net.state_dict()[name]
            )
        self.client_grads[index] = local_grad
