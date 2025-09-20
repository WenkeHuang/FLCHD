import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
import numpy as np

import wandb
from backbone.chdmodule.modules.metrics import CHDMetricCLSBinary, CHDMetricCLSMulticlass
from backbone.chdmodule.utilities.enums import Stage
from models.utils.federated_model import FederatedModel
from utils.args import *
from datasets.chd import CHDImageDataItem
from utils.constract import DiagColType


class WPOptim(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, alpha=0.05, **kwargs):
        defaults = dict(alpha=alpha, **kwargs)
        super(WPOptim, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def generate_delta(self, zero_grad=False):
        device = self.param_groups[0]["params"][0].device
        grad_norm = torch.norm(
            torch.stack([
                (1.0 * p.grad).norm(p=2).to(device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None]),p=2
        )
        for group in self.param_groups:
            scale = group["alpha"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                delta = 1.0 * p.grad * scale.to(p)
                p.add_(delta)
                self.state[p]["delta"] = delta

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["delta"])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()


class HarmoFL(FederatedModel):
    NAME = 'harmofl'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(HarmoFL, self).__init__(nets_list, args, transform)
        self.chdtype = args.chdtype
        self.alpha = getattr(args, 'alpha', 0.05)  
        self.optimizers = [] 

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

        self.optimizers = []
        for _ in range(len(self.nets_list)):
            self.optimizers.append(None)  

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = total_clients
        self.online_clients = online_clients

        print('Using online_clients:', online_clients)

        for i in online_clients:
            self._train_net_with_perturbation(i, self.nets_list[i], priloader_list[i])

        self._aggregate_with_harmofl()

        return None

    def _train_net_with_perturbation(self, index, net: nn.Module, train_loader):
        net = net.to(self.device)
        net.train()

        optimizer = WPOptim(
            params=net.parameters(),
            base_optimizer=optim.AdamW,
            alpha=self.alpha,
            lr=1e-4,
            weight_decay=1e-4
        )
        self.optimizers[index] = optimizer

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer.base_optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            threshold=1e-4,
            threshold_mode='abs',
            cooldown=0,
            min_lr=1e-6
        )

        if self.chdtype == 'binary':
            print(f"Client {index} using chd type binary")
            metric_train = CHDMetricCLSBinary(stage=f"{Stage.TRAIN}").to(self.device)
        else:
            print(f"Client {index} using chd type multi-class")
            metric_train = CHDMetricCLSMulticlass(
                DiagColType,
                stage=f"{Stage.TRAIN}"
            ).to(self.device)

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
                logits = step_output.model_output['logits'].to(self.device)
                metric_train.update(logits, labels)

                optimizer.zero_grad()
                loss.backward()

                optimizer.generate_delta(zero_grad=True)

                step_output_perturbed = net.model.shared_step(gpu_batch, batch_idx)
                loss_perturbed = step_output_perturbed.loss

                loss_perturbed.backward()

                optimizer.step(zero_grad=True)

                iterator.desc = f"Local Participant {index} loss = {loss.item():.3f}"

            metrics_results = metric_train.compute()
            metric_train.reset()

            print(f"Local Participant {index} Epoch {epoch + 1}/{self.local_epoch} metrics results:")
            print('Train/CLS/ACC:', f"{metrics_results['Train/CLS/ACC'].item():.4f}    ",
                  'Train/CLS/AUC:', f"{metrics_results['Train/CLS/AUC'].item():.4f}    ",
                  'Train/CLS/F1:', f"{metrics_results['Train/CLS/F1'].item():.4f}")

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

            scheduler.step(loss)

    def _aggregate_with_harmofl(self):
        """Aggregate models with HarmoFL communication strategy"""
        with torch.no_grad():
            client_num = len(self.online_clients)
            client_weights = [1.0 / client_num for _ in range(client_num)]

            global_w = self.global_net.state_dict()
            for key in global_w.keys():
                temp = torch.zeros_like(global_w[key])
                for idx, client_idx in enumerate(self.online_clients):
                    temp += client_weights[idx] * self.nets_list[client_idx].state_dict()[key]
                global_w[key].data.copy_(temp)

                for client_idx in self.online_clients:
                    self.nets_list[client_idx].state_dict()[key].data.copy_(global_w[key])

            self.global_net.load_state_dict(global_w)