from typing import List, Any

import pytorch_lightning as pl
import torch
import torchvision as tv
from pytorch_lightning import metrics
from torch import Tensor, nn
from torch.optim import lr_scheduler


class ClsModule(pl.LightningModule):

    def __init__(
            self,
            num_classes=2
    ):
        super().__init__()
        self.num_classes = num_classes

        mobilenet = tv.models.mobilenet_v2(pretrained=True)
        in_linear = mobilenet.classifier[1].in_features
        mobilenet.classifier = nn.Sequential(
            nn.Dropout2d(),
            nn.Linear(in_linear, num_classes)
        )
        self.fc_activation = nn.Softmax(dim=1)

        self.train_metrics = nn.ModuleDict({
            'accuracy': metrics.Accuracy(),
            'F1': metrics.F1(),
        })
        self.val_metrics = nn.ModuleDict({
            'accuracy': metrics.Accuracy(),
            'F1': metrics.F1(),
            'cm': metrics.ConfusionMatrix(num_classes)
        })

        self.classifier = mobilenet
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: Tensor):
        x = self.classifier(x)
        return self.fc_activation(x)

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.classifier(x)

        loss = self.criterion(y_pred, y_true)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        for name, metric in self.train_metrics.items():
            value = metric(y_pred, y_true)
            self.log(f'train_{name}', value, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.classifier(x)

        loss = self.criterion(y_pred, y_true)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        for name, metric in self.val_metrics.items():
            value = metric(y_pred, y_true)
            if len(value.shape) == 0:
                self.log(f'val_{name}', value, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        cm = self.val_metrics['cm'].compute()
        print(cm)

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.classifier(x)

        loss = self.criterion(y_pred, y_true)

        self.log('test_loss', loss, prog_bar=True, on_epoch=True)
        for name, metric in self.val_metrics.items():
            value = metric(y_pred, y_true)
            if len(value.shape) == 0:
                self.log(f'test_{name}', value, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def test_epoch_end(self, outputs: List[Any]) -> None:
        cm = self.val_metrics['cm'].compute()
        print(cm)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        return [optimizer], [scheduler]
