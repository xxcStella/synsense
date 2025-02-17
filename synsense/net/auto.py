"""
This file contrains class for train and test on PC with Sinabs framework.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold

import sinabs
import sinabs.layers as sl

from sinabs.from_torch import from_model

class AutoSNN():
    """
    This class contains training and testing methods for SNN. The key thing to 
    provide is an ANN structure.

    Params:
        batch_size: int. batch size for training.
    """
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
    
    def _single_train(
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
            print(f"K fold: {fold}")

            # Transfer ANN to SNN using Sinabs function from_model
            snn = from_model(
                model=model,
                input_shape=input_shape,  # (C, H, W)
                # batch_size=self.batch_size, # 不加此参数,由num_timesteps自动计算,后续batch大小可变
                num_timesteps=timesteps, #(T)
                add_spiking_output=True
            ).to(device)

            optimizer = optimizer_class(snn.parameters(), init_lr)
            loss_fn   = loss_fn_class()
            scheduler = scheduler_class(optimizer, step_size=step_size, gamma=gamma)

            # Split the train set into training and validation sets
            train_sub = Subset(train_dataset, train_idx)
            val_sub   = Subset(train_dataset, val_idx)
            train_loader = DataLoader(train_sub, batch_size=self.batch_size, shuffle=True, drop_last=True)
            val_loader   = DataLoader(val_sub, batch_size=self.batch_size, drop_last=True)

            # Train current fold
            max_acc, best_snn = self._single_train(
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

        print(f"mean train acc: {sum(fold_train_acc)/len(fold_train_acc) :.2f}"
              f" mean val acc: {sum(fold_test_acc)/len(fold_test_acc) :.2f}")
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
        snn = torch.load(model_path).to(device)
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
