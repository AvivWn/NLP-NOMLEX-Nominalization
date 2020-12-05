import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
seed_everything(0)
class CoolSystem(pl.LightningModule):
	def __init__(self, classes=10):
		super().__init__()
		self.save_hyperparameters()
		# not the best model...
		self.l1 = torch.nn.Linear(28 * 28, self.hparams.classes)

	def forward(self, x):
		return torch.relu(self.l1(x.view(x.size(0), -1)))

	# def prepare_data(self):
		# self.mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
		# self.mnist_val = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

	def setup(self, stage):
		self.mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
		self.mnist_val = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

	def train_dataloader(self):
		return DataLoader(self.mnist_train, batch_size=32, num_workers=4)

	def val_dataloader(self):
		return DataLoader(self.mnist_val, batch_size=32, num_workers=4)

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		loss = F.cross_entropy(y_hat, y)
		tensorboard_logs = {'train_loss': loss}
		return {'loss': loss, 'log': tensorboard_logs}

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		loss = F.cross_entropy(y_hat, y)
		return {'val_loss': loss}

	def validation_epoch_end(self, outputs):
		avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
		return {'val_loss': avg_loss}

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=0.001)

def debug():
	# data
	# mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
	# mnist_train = DataLoader(mnist_train, batch_size=32, num_workers=4)
	# mnist_val = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
	# mnist_val = DataLoader(mnist_val, batch_size=32, num_workers=4)

	# model
	model = CoolSystem()

	# most basic trainer, uses good defaults
	trainer = Trainer(progress_bar_refresh_rate=20, max_epochs=10, distributed_backend="dp", gpus=-1)
	trainer.fit(model) #, mnist_train, mnist_val)

if __name__ == "__main__":
	debug()