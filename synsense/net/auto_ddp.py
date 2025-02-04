"""
This file contrains class for train and test on PC with Sinabs framework.
This is a experimental file for multi GPU training.
"""
import os
import copy
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Subset, DistributedSampler
from sklearn.model_selection import KFold

import sinabs
import sinabs.layers as sl
from sinabs.from_torch import from_model

def ddp_setup(rank: int, world_size: int, backend: str = "nccl"):
    """
    初始化分布式进程组。在 GPU 环境下一般用 nccl 作为后端。
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    # 让当前进程只使用 rank 对应的那张 GPU
    torch.cuda.set_device(rank)

def cleanup():
    """ 销毁进程组，在训练结束后调用。 """
    dist.destroy_process_group()

class AutoSNN():
    """
    This class contains training and testing methods for SNN. The key thing to 
    provide is an ANN structure.

    Params:
        batch_size: int. batch size for training.
    """
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def _single_train_single(
            self,
            model_path: str,
            train_loader: DataLoader,
            epochs: int,
            snn: sinabs.network.Network,
            device: str,
            timesteps: int,
            optimizer: Optional[torch.optim.Optimizer],
            loss_fn: Optional[nn.Module],
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
            _past_Kfold_max_acc: float
        ) -> Tuple[float, sinabs.network.Network]:
        """
        Train a model without K-fold cross validation. The model with the highest accuracy
        will be saved in path given when initilizing the class.

        Params:
            _past_Kfold_max_acc (float): record past K-Folds highest accuracy. Automatically\
                        obtain from the "train" function. You don't need to fill it.

        Return:
            max_acc (float): maximum classification accuracy in these epochs.
            best_snn (sinabs.network.Network): best snn in the whole single Fold.
        """
        snn.train()
        max_acc = 0.0
        best_snn = None
        for e in range(epochs):
            running_loss = 0.0
            cls_true_num = 0
            total_num = 0

            for _, (input, target) in enumerate(train_loader): # different batches
                optimizer.zero_grad()
                snn.reset_states()

                input = input.to(device)
                input = sl.FlattenTime()(input)
                target = target.to(device)

                output = snn(input)
                output = output.view(self.batch_size, timesteps, -1) # (N, T, Class)
                sum_output = output.sum(1)  # (N, Class)

                loss = loss_fn(sum_output, target)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    running_loss += loss
                    cls_true_num += torch.eq(sum_output.argmax(dim=1), target).sum()
                    total_num += self.batch_size

            scheduler.step()    # update learning rate

            # If reaches highest accuracy in this epoch, update max_acc and
            # save the model to best_snn to evaluate.
            current_acc = cls_true_num / total_num
            if current_acc > max_acc: 
                max_acc = current_acc
                best_snn = snn
                # If reaches highest acc in the whole past kFolds, save
                # the model to disk.
                if current_acc >= _past_Kfold_max_acc:
                    torch.save(snn, model_path)

            print(
                f"epoch: {e}; Accuracy: {current_acc*100: .2f}%;"
                f" Loss: {running_loss: .2f}; current_lr: {scheduler.get_last_lr()[0]: .6f}"
            )

        # return maxium accuracy during the whole epochs
        return max_acc, best_snn
    
    def _single_train_ddp(
            self,
            rank: int,
            model_path: str,
            train_loader: DataLoader,
            epochs: int,
            snn: sinabs.network.Network,
            device: str,
            timesteps: int,
            optimizer: Optional[torch.optim.Optimizer],
            loss_fn: Optional[nn.Module],
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
            _past_Kfold_max_acc: float
        ) -> Tuple[float, sinabs.network.Network]:
        """
        Multi GPU version of the _single_train_single function.
        Train a model without K-fold cross validation. The model with the highest accuracy
        will be saved in path given when initilizing the class.

        Params:
            rank (int): current processing id. single machine means current GPU id. 
            _past_Kfold_max_acc (float): record past K-Folds highest accuracy. Automatically\
                        obtain from the "train" function. You don't need to fill it.

        Return:
            max_acc (float): maximum classification accuracy in these epochs.
            best_snn (sinabs.network.Network): best snn in the whole single Fold.
        """
        snn.train()
        max_acc = 0.0
        best_snn = None
        for e in range(epochs):
            # 如果使用分布式 sampler，需要在每个 epoch 开始时 set_epoch
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(e)

            running_loss = 0.0
            cls_true_num = 0
            total_num = 0

            for _, (input, target) in enumerate(train_loader): # different batches
                optimizer.zero_grad()
                snn.module.reset_states()

                input = input.to(device)
                input = sl.FlattenTime()(input)
                target = target.to(device)

                output = snn(input)
                output = output.view(self.batch_size, timesteps, -1) # (N, T, Class)
                sum_output = output.sum(1)  # (N, Class)

                loss = loss_fn(sum_output, target)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    running_loss += loss
                    cls_true_num += torch.eq(sum_output.argmax(dim=1), target).sum()
                    total_num += self.batch_size

            scheduler.step()    # update learning rate

            # --- 在多卡场景下，需要汇总各卡的 cls_true_num 和 total_num ---
            # 让所有 rank 的统计量做 all_reduce 求和
            # 这样算出来的才是全局的 total_num 和正确数之和。
            cls_true_num = cls_true_num.to(device)
            total_num = torch.tensor(total_num, device=device, dtype=cls_true_num.dtype)
            dist.all_reduce(cls_true_num, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_num,   op=dist.ReduceOp.SUM)

            current_acc = cls_true_num / total_num

            # loss 同理也可做一个 all_reduce，然后只在 rank=0 打印
            running_loss = running_loss.to(device)
            dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)

            # 只让 rank=0 做打印、模型保存
            if rank == 0:
                if current_acc > max_acc: 
                    max_acc = current_acc
                    best_snn = copy.deepcopy(snn.module)
                    # If reaches highest acc in the whole past kFolds, save
                    # the model to disk.
                    if current_acc >= _past_Kfold_max_acc:
                        torch.save(snn.module, model_path)

                print(
                    f"epoch: {e}; Accuracy: {current_acc*100: .2f}%;"
                    f" Loss: {running_loss: .2f}; current_lr: {scheduler.get_last_lr()[0]: .6f}"
                )

        # return maxium accuracy during the whole epochs
        return max_acc, best_snn
    
    def _single_validation(
            self,
            val_loader: DataLoader,
            snn: sinabs.network.Network,
            device: str,
            timesteps: int,
        ) -> float:
        """
        Calculate the validation set accuracy.

        Return:
            float. validation accuracy.
        """
        cls_true_num = 0
        total_num = 0
        for _, (input, target) in enumerate(val_loader):
            with torch.no_grad():
                snn.reset_states()

                input = input.to(device)
                input = sl.FlattenTime()(input)
                target = target.to(device)

                output = snn(input)
                output = output.view(self.batch_size, timesteps, -1) # (N, T, Class)
                sum_output = output.sum(1)  # (N, Class)

                cls_true_num += torch.eq(sum_output.argmax(dim=1), target).sum()
                total_num += self.batch_size

        val_acc = cls_true_num / total_num
        print(f"val accuray: {val_acc * 100: .2f}%")
        return val_acc

    def train(
            self,
            model_path: str,
            model: nn.Sequential,
            train_dataset: Dataset,
            device: str,

            k_folds: int,
            epochs: int,
            optimizer_class: Optional[torch.optim.Optimizer]=torch.optim.Adam,
            loss_fn_class: Optional[nn.Module]=nn.CrossEntropyLoss,

            init_lr: Optional[float]=1e-3,
            scheduler_class: Optional[torch.optim.lr_scheduler._LRScheduler]=torch.optim.lr_scheduler.StepLR,
            step_size: Optional[int]=6,
            gamma: Optional[float]=0.1,

            is_ddp: bool=False,    # 增加一个标志，是否使用DDP
            rank: int=0,           # 当前进程的 rank
            world_size: int=1,      # 总进程数

            kwargs: Optional[dict]={}
        ) -> Tuple[list, list]:
        """
        Auto training function with K-fold cross validation.

        Params:
            model_path (str): model storing path.
            model (Sequential): your inital ANN.
            train_dataset (Dataset): Dataset_Texture_Stream class. (T, C, H, W)
            device (str): cpu or cuda.
            k_folds (int): k fold cross validation. min: 2.
            epochs (int): total epochs in 1 single training.
            optimizer (optim): pytorch optimizer, default: Adam.
            loss_fn (Module): default: CrossEntropy.
            init_lr (float): initial learning rate, default: 1e-3.
            scheduler: used to adjust the learning rate, default: StepLR.
            step_size (int): change learning rate every step_size steps.
            gamma (float): devide current learning rate by gamma when changing lr.
            is_ddp (bool): whether to use DDP. default=False.
            rank (int): current processing id. single machine means current GPU id.
            world_size (int): total processing number. single machine means total GPU number.

        Return:
            fold_train_acc, fold_test_acc (list): contain accuracy result in k folds.
        """
        
        kfold = KFold(n_splits=k_folds, shuffle=True)

        timesteps = train_dataset[0][0].shape[0]    #(T)
        input_shape = train_dataset[0][0].shape[1:] # (C, H, W)

        fold_train_acc = []
        fold_test_acc  = []
        _past_Kfold_max_acc = 0.0 # record the max acc in past folds

        # k fold cross validation loop
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
            if rank == 0:
                print(f"=========== K fold: {fold} ===========")

            # Transfer ANN to SNN using Sinabs function from_model
            snn = from_model(
                model=model,
                input_shape=input_shape,  # (C, H, W)
                # batch_size=self.batch_size, # 不加此参数,由num_timesteps自动计算,后续batch大小可变
                num_timesteps=timesteps, #(T)
                add_spiking_output=True,
                **kwargs
            ).to(device)

            optimizer = optimizer_class(snn.parameters(), init_lr)
            loss_fn   = loss_fn_class()
            scheduler = scheduler_class(optimizer, step_size=step_size, gamma=gamma)

            # Split the train set into training and validation sets
            train_sub = Subset(train_dataset, train_idx)
            val_sub   = Subset(train_dataset, val_idx)

            # 分布式训练,使用DDP
            if is_ddp:
                # 分布式场景下使用 DistributedSampler
                train_sampler = DistributedSampler(train_sub, num_replicas=world_size, rank=rank, shuffle=True)
                train_loader = DataLoader(train_sub, batch_size=self.batch_size, sampler=train_sampler, drop_last=True)
                
                # 验证集也可以用分布式Sampler或普通Sampler。一般验证集也可以分布式测，但要汇总。
                val_loader = DataLoader(val_sub, batch_size=self.batch_size, drop_last=True)
                
                # 用 DDP 包装 SNN
                ddp_snn = DDP(snn, device_ids=[rank], output_device=rank, find_unused_parameters=True)

                # Train current fold
                max_acc, best_snn = self._single_train_ddp(
                        rank=rank,
                        model_path=model_path,
                        train_loader=train_loader,
                        epochs=epochs,
                        snn=ddp_snn,
                        device=device,
                        timesteps=timesteps,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        scheduler=scheduler,
                        _past_Kfold_max_acc=_past_Kfold_max_acc
                    )
                # 如果在 rank=0 上，就更新一下全局最优
                if rank == 0:
                    fold_train_acc.append(max_acc.item())
                    if max_acc >= _past_Kfold_max_acc:
                        _past_Kfold_max_acc = max_acc

                # Validate current fold
                if rank == 0:
                    val_acc = self._single_validation(
                            val_loader=val_loader,
                            snn=best_snn,
                            device=device,
                            timesteps=timesteps
                        )
                    fold_test_acc.append(val_acc.item())
            # 单GPU版本
            else:
                train_loader = DataLoader(train_sub, batch_size=self.batch_size, shuffle=True, drop_last=True)
                val_loader   = DataLoader(val_sub, batch_size=self.batch_size, drop_last=True)

                # Train current fold
                max_acc, best_snn = self._single_train_single(
                        model_path=model_path,
                        train_loader=train_loader,
                        epochs=epochs,
                        snn=snn,
                        device=device,
                        timesteps=timesteps,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        scheduler=scheduler,
                        _past_Kfold_max_acc=_past_Kfold_max_acc
                    )
                fold_train_acc.append(max_acc.item())
                if max_acc >= _past_Kfold_max_acc:
                    _past_Kfold_max_acc = max_acc

                # Validate current fold
                val_acc = self._single_validation(
                        val_loader=val_loader,
                        snn=best_snn,
                        device=device,
                        timesteps=timesteps
                    )
                fold_test_acc.append(val_acc.item())

        if rank == 0:
            print(f"mean train acc: {sum(fold_train_acc)/len(fold_train_acc) :.4f}"
                f" mean val acc: {sum(fold_test_acc)/len(fold_test_acc) :.4f}")
        return fold_train_acc, fold_test_acc

    def test(
            self,
            test_dataset: Dataset,
            model_path: str,
            device: str
        ) -> None:
        """
        Test on testing set.

        Params:
            test_dataset: Dataset_Texture_Stream class. (T, C, H, W)
            model_path (str): the path where your model stores.
            device (str): cpu or cuda.
        """
        snn = torch.load(model_path)
        timesteps = test_dataset[0][0].shape[0] 
        test_loader = DataLoader(test_dataset, self.batch_size, drop_last=True)

        cls_true_num = 0
        total_num = 0

        snn.eval()
        for _, (input, target) in enumerate(test_loader):
            with torch.no_grad():
                snn.reset_states()

                input = input.to(device)
                input = sl.FlattenTime()(input)
                target = target.to(device)

                output = snn(input)
                output = output.view(self.batch_size, timesteps, -1) # (N, T, Class)
                sum_output = output.sum(1)  # (N, Class)

                cls_true_num += torch.eq(sum_output.argmax(dim=1), target).sum()
                total_num += self.batch_size

        print(f"test accuracy: {(cls_true_num / total_num) * 100: .2f}%")
