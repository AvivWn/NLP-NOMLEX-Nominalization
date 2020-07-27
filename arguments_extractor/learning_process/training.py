import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from arguments_extractor import config

MAX_EPOCHS = 20

def ids_to_labels(ids_list, tagset):
	return [tagset[i] for i in ids_list]

def batch_to_device(batch_tuple, device):
	if device == "cpu":
		return batch_tuple

	batch_in_device = []
	for x in batch_tuple:
		batch_in_device.append(x.to(device))

	return batch_in_device

def test(test_dataloader: DataLoader, tagset: dict, model, device):
	model.eval()

	pred_labels = []
	true_labels = []

	# Validation on the given test-set
	for step, batch in tqdm(enumerate(test_dataloader), "Validation", leave=False):
		# Transfer to the device
		batch_in_device = batch_to_device(batch, device)
		batch_inputs = batch_in_device[:-1]
		batch_labels = batch[-1]

		# Model computations- forward
		with torch.no_grad():
			logits = model.forward(*batch_inputs)

		preds = logits.argmax(dim=1, keepdim=True).squeeze(1).detach().cpu().numpy()
		pred_labels += ids_to_labels(preds, tagset)
		true_labels += ids_to_labels(batch_labels.numpy().astype(int), tagset)

	print(classification_report(true_labels, pred_labels))

def train(train_dataloader: DataLoader, test_dataloader: DataLoader, tagset: dict, model, optimizer, model_path, device):
	# Epoch loop
	for epoch in range(MAX_EPOCHS):
		model.train()
		total_loss = 0

		# Training on every batch in the given test-set
		for step, batch in tqdm(enumerate(train_dataloader), f"Training (EPOCH={epoch})"):
			# Transfer to the device
			batch_in_device = batch_to_device(batch, device)
			batch_inputs = batch_in_device[:-1]
			batch_labels = batch_in_device[-1]

			# Model computations- forward
			optimizer.zero_grad()
			logits = model.forward(*batch_inputs)

			# Backward
			loss = F.nll_loss(logits, batch_labels).mean()
			total_loss += loss.detach().cpu().numpy()
			loss.backward()
			optimizer.step()

		print(f"Average Loss: {round(total_loss/float(len(train_dataloader)), 3)}")

		# Validation test

		if config.DEBUG:
			test(train_dataloader, tagset, model, device)

		test(test_dataloader, tagset, model, device)

		# Save the model state
		if isinstance(model, torch.nn.DataParallel):
			torch.save(model.module.state_dict(), model_path)
		else:
			torch.save(model.state_dict(), model_path)

def training(train_dataloader: DataLoader, test_dataloader: DataLoader, tagset: dict, model, optimizer, model_path):
	# CUDA for PyTorch
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")

	if use_cuda:
		model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

	model.to(device)
	train(train_dataloader, test_dataloader, tagset, model, optimizer, model_path, device)