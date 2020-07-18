import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np

class MyPytorchModel(pl.LightningModule):
    
    def __init__(self, hparams, input_size=3 * 32 * 32, num_classes=10, dropout_p=0):
        super().__init__()

        # set hyperparams
        self.hparams = hparams

        ########################################################################
        # TODO: Initialize your model!                                         #
        ########################################################################


        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, num_classes)
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
       
        # x.shape = [batch_size, 3, 32, 32] -> flatten the image first
        x = x.view(x.shape[0], -1)

        # feed x into model!
        x = self.model(x)

        return x
    
    def general_step(self, batch, batch_idx, mode):
        images, targets = batch

        # forward pass
        out = self.forward(images)

        # loss
        loss = F.cross_entropy(out, targets)

        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()
        return loss, n_correct
    
    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.dataset[mode])
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        return {'loss': loss, 'train_n_correct':n_correct, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_n_correct':n_correct}
    
    def test_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct':n_correct}

    def validation_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        #print("Val-Acc={}".format(acc))
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'val_acc': acc, 'log': tensorboard_logs}

    def prepare_data(self):

        # create dataset
        CIFAR_ROOT = "../datasets/cifar10"
        ########################################################################
        # TODO: Define your transforms (convert to tensors, normalize).        #
        # If you want, you can also perform data augmentation!                 #
        ########################################################################

        mean=[0.49191375, 0.48235852, 0.44673872]
        std=[0.24706447, 0.24346213, 0.26147554]
        
        my_transform = transforms.Compose([
#             transforms.ColorJitter(brightness=0.3, contrast=0.6, saturation=0.5),
#             transforms.RandomAffine(degrees=(10,90), translate=(0.1,0.3), shear=0.01),
#             transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std, inplace=False)
        ])

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        cifar_complete = torchvision.datasets.ImageFolder(root=CIFAR_ROOT, transform=my_transform)
        torch.manual_seed(0)
        N = len(cifar_complete)
        cifar_train, cifar_val, cifar_test = torch.utils.data.random_split(cifar_complete, 
                                                                           [int(N*0.6), int(N*0.2), int(N*0.2)])
        torch.manual_seed(torch.initial_seed())

        # assign to use in dataloaders
        self.dataset = {}
        self.dataset["train"], self.dataset["val"], self.dataset["test"] = cifar_train, cifar_val, cifar_test

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], shuffle=True, batch_size=self.hparams["batch_size"])

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=self.hparams["batch_size"])
    
    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.hparams["batch_size"])

    def configure_optimizers(self):

        optim = None
        ########################################################################
        # TODO: Define your optimizer.                                         #
        ########################################################################
        
        optim = torch.optim.Adam(self.model.parameters(), self.hparams["learning_rate"])
  
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return optim

    def getTestAcc(self, loader = None):
        self.model.eval()
        self.model = self.model.to(self.device)

        if not loader: loader = self.test_dataloader()

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc