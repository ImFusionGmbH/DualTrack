from abc import ABC, abstractmethod
from typing import Callable, Protocol, runtime_checkable
from torch import nn
import torch


class BaseTrackingEstimator(nn.Module, ABC): 

    @abstractmethod
    def predict(self, batch) -> torch.Tensor:
        ...

    @abstractmethod
    def get_loss(self, batch, pred=None):
        ...

    @property
    def device(self): 
        return next(self.parameters()).device


class LocalEncoderTrackingEstimator(BaseTrackingEstimator): 
    def predict(self, batch) -> torch.Tensor:
        if "images" in batch:
            images = batch["images"].to(self.device)
            B, C, N, H, W = images.shape
            # targets = batch["targets"].to(args.device)
            outputs = self(images)
        else:
            outputs = self(batch["image_features"].to(self.device))
        return outputs

    def get_loss(self, batch, pred=None):
        if pred is None:
            outputs = self.predict(batch)
        else:
            outputs = pred
        targets = batch["targets"].to(self.device)
        loss = torch.nn.functional.mse_loss(outputs, targets, reduction="none").mean()
        return loss

