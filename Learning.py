import operator
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from collections import defaultdict
#from matplotlib import pyplot as plt

import CreateData
from DictsAndTables import get_subentries_table
from Model import tagging_model, scoring_model

# Constants
tags_dict = dict([(tag.upper(), i + 4) for i, (tag, _, _) in enumerate(get_subentries_table())] + [("NONE", 0), ("NOM", 1), ("NOM_SUBJECT", 2), ("NOM_OBJECT", 3)])
BATCH_SIZE = 64
TRAIN_LIMIT_SIZE = 10000
TEST_LIMIT_SIZE = 10000
MODEL_NUM = 2

# Hyper Parameters
num_of_epochs = 10
learning_rate = 0.001
momentum_factor = 0.75

tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertTokenizer', 'bert-base-cased', do_basic_tokenize=False)

model_name = "tagging_model" if MODEL_NUM == 1 else "scoring_model"
trained_model_filename = CreateData.LEARNING_FILES_LOCATION + "trained_" + model_name

show_progress_bar = True



def encode_tags(tags):
	tags_idx = []

	for i in range(len(tags)):
		tags_idx.append(tags_dict.get(tags[i], -1))

	return tags_idx

def raw_to_batch_examples(sent, splitted_tags_idxs, sent_idx):
	batch_examples = []

	right_tags_list = splitted_tags_idxs["+"]

	if len(right_tags_list) == 0:
		return []

	right_tags_idxs = splitted_tags_idxs["+"][0]
	wrong_tags_idxs_list = splitted_tags_idxs["-"]
	splitted_sent = sent.split(" ")

	if MODEL_NUM == 1:
		batch_examples.append((sent_idx, splitted_sent, right_tags_idxs, len(splitted_sent), True))

	else: #if MODEL_NUM == 2:
		batch_examples.append((sent_idx, splitted_sent, right_tags_idxs, len(splitted_sent), True))

		for wrong_tags_idxs in wrong_tags_idxs_list:
			batch_examples.append((sent_idx, splitted_sent, wrong_tags_idxs, len(splitted_sent), False))

	return batch_examples

def forward(batch_examples, model, device):
	max_length = max(batch_examples, key=operator.itemgetter(3))[3]

	# Sorting by sentence length first and then by sentence index
	list.sort(batch_examples, key=operator.itemgetter(3, 0, 4), reverse=True)

	# Adding padding to the sentences in the batch
	padded_batch_tags = []
	padded_batch_indexed_tokens = []
	sents_lengths = []
	for _, splitted_sent, tags_idxs, sent_length, _ in batch_examples:
		padding_length = max_length - sent_length
		padded_batch_tags.append(tags_idxs + [tags_dict.get("NONE", -1)] * padding_length)

		sents_lengths.append(sent_length)
		padded_batch_sent = splitted_sent + ["<PAD>"] * padding_length
		padded_batch_indexed_tokens.append(tokenizer.convert_tokens_to_ids(padded_batch_sent))

	# Transform the inputs into tensors, before give it to the model
	padded_batch_indexed_tokens = torch.tensor(padded_batch_indexed_tokens).to(device)
	sents_lengths = torch.tensor(sents_lengths).to(device)

	batch_losses = []

	# Calculating loss differenty for each model
	if MODEL_NUM == 1:
		# Calling the model to get the output
		outputs = model(padded_batch_indexed_tokens, sents_lengths).permute(0, 2, 1)

		# Calculating the loss
		padded_batch_tags = torch.Tensor(padded_batch_tags).long().to(device)
		batch_losses.append(torch.nn.functional.nll_loss(outputs, padded_batch_tags))
	else: #if MODEL_NUM == 2:
		# Calling the model to get the output
		outputs = model(padded_batch_indexed_tokens, torch.tensor(padded_batch_tags).to(device), sents_lengths)

		right_score = None
		other_best_score = None
		last_sent_idx = -1

		output_by_sents = []
		curr_outputs_for_sent = []

		for ex_idx in range(len(batch_examples)):
			sent_idx, splitted_sent, tags_idx, sent_length, is_right_tags = batch_examples[ex_idx]

			if last_sent_idx == -1:
				last_sent_idx = sent_idx

			# Is it a new sentence, or the last one?
			if sent_idx != last_sent_idx:
				if type(other_best_score) == torch.Tensor and type(right_score) == torch.Tensor:
					loss = torch.Tensor.max(torch.Tensor([0]).to(device),
											torch.Tensor([1]).to(device) - (right_score - other_best_score))
					batch_losses.append(loss)

				output_by_sents.append((curr_outputs_for_sent, right_score))

				last_sent_idx = sent_idx
				right_score = None
				other_best_score = None
				curr_outputs_for_sent = []

			curr_outputs_for_sent.append(outputs[ex_idx])

			if is_right_tags:
				right_score = outputs[ex_idx]
			else:
				if type(other_best_score) != torch.Tensor or \
						not torch.Tensor.equal(torch.Tensor.max(torch.Tensor([0]).to(device),
																outputs[ex_idx] - other_best_score[0]), # TODO
											   torch.Tensor([0]).to(device)):
					other_best_score = outputs[ex_idx]

		outputs = output_by_sents

	return batch_losses, outputs

def get_acc(batch_examples, outputs):
	batch_acc = 0

	if MODEL_NUM == 1:
		preds = outputs.argmax(dim=1, keepdim=True).squeeze(dim=1)  # get the index of the max log-probability
		preds = preds.cpu().numpy()

		for i in range(len(preds)):
			count_right = 0

			for j in range(len(batch_examples[i][2])):
				if preds[i][j] == batch_examples[i][2][j]:
					count_right += 1

			batch_acc += count_right / len(batch_examples[i][2])

		batch_acc /= len(batch_examples)
	else: # if MODEL_NUM == 2:
		count_right = 0

		for (curr_outputs_for_sent, right_score) in outputs:
			if max(curr_outputs_for_sent) == right_score:
				count_right += 1

		batch_acc = count_right / len(outputs)

	return batch_acc


def train(model, device, train_dataset, optimizer, epoch):
	"""
	Trains the model for a single epoch
	:param model: the model
	:param device: the device where the computation will be on (cpu or cuda)
	:param train_dataset: the training data
	:param optimizer: the optimizer that will help training the model
	:param epoch: the number of the current epoch
	:return: None
	"""

	model.train()

	random_indexes = np.arange(len(train_dataset))[:TRAIN_LIMIT_SIZE]
	np.random.shuffle(random_indexes)
	sent_count = 0
	batch_examples = []

	default_desc = 'Epoch ' + str(epoch) + ' [Training]'
	iterable = tqdm(random_indexes, desc=default_desc, leave=False) if show_progress_bar else random_indexes

	for sent_idx in iterable:
		sent, splitted_tags_idxs = train_dataset[sent_idx]
		sent_count += 1 # Counting the sentences (for progress printing)

		# Aggregating the batch examples
		batch_examples += raw_to_batch_examples(sent, splitted_tags_idxs, sent_idx)

		if len(batch_examples) >= BATCH_SIZE:
			optimizer.zero_grad()

			# Calculatin the losses of the examples in the batch
			batch_losses, _ = forward(batch_examples, model, device)

			# If there was any positive loss, propogate it and update the model
			if batch_losses != []:
				loss = sum(batch_losses).to(device) / len(batch_losses)
				loss.backward()
				optimizer.step()

				if show_progress_bar:
					iterable.set_description(default_desc + ' - Loss = {:.6f}'.format(loss.item()))
					iterable.refresh()
				else:
					print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
						epoch, sent_count, len(random_indexes),
						100. * sent_count / len(random_indexes), loss.item()))

			batch_examples = []

def valid(model, device, test_dataset, dataset_name, epoch):
	"""
	Checks the prediction of the model over the validaiton data
	:param model: the trained model
	:param device: the device where the computation will be on (cpu or cuda)
	:param test_dataset: the validation data
	:param dataset_name: the name of the dataset (for printing)
	:return: loss, acc score
	"""

	model.eval()
	test_loss = 0
	test_acc = 0

	random_indexes = np.arange(len(test_dataset))[:TEST_LIMIT_SIZE]
	np.random.shuffle(random_indexes)
	sent_count = 0
	batch_examples = []
	num_of_batches = 0

	default_desc = 'Epoch ' + str(epoch) + ' [' + dataset_name + ' test]'
	iterable = tqdm(random_indexes, desc=default_desc, leave=False) if show_progress_bar else random_indexes

	with torch.no_grad():
		for sent_idx in iterable:
			sent, tags_dict = test_dataset[sent_idx]
			sent_count += 1  # Counting the sentences (for progress printing)

			# Aggregating the batch examples
			batch_examples += raw_to_batch_examples(sent, tags_dict, sent_idx)

			if len(batch_examples) >= BATCH_SIZE:
				num_of_batches += 1

				# Calculatin the losses of the examples in the batch
				batch_losses, outputs = forward(batch_examples, model, device)

				# If there was any positive loss, propogate it and update the model
				if batch_losses != []:
					loss = sum(batch_losses).to(device) / len(batch_losses)

					# Sum up current loss
					current_loss = loss.item()
					test_loss += current_loss

					# Calculate the accuracy of the batch
					current_acc = get_acc(batch_examples, outputs)
					test_acc += current_acc

					if show_progress_bar:
						iterable.set_description(default_desc + ' - loss = {:.6f}, ACC = {:.3f}'.format(loss.item(), current_acc))
						iterable.refresh()

				batch_examples = []

	test_loss /= num_of_batches
	test_acc /= num_of_batches

	return test_loss, test_acc

def test(model, device, test_dataset):
	"""
	Predicts over the test set, and writing the results in a suitable file
	:param model: the trained model
	:param device: the device where the computation will be on (cpu or cuda)
	:param test_dataset: the test set
	:return: None
	"""

	model.eval()

	predictions = []

	with torch.no_grad():
		for sent in test_dataset:
			output = model(sent, device)

			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			predictions.append(pred)

	return predictions


def text_to_tags(sentence, arguments_as_text):
	splitted_sent = sentence.split(" ")
	tags = ["NONE"] * len(splitted_sent)

	argumets = arguments_as_text.split(" ")

	for x in argumets:
		arg = x.split("_")
		tag = "_".join(arg[:-2])
		start_idx = int(arg[-2])
		end_idx = int(arg[-1])
		idx = start_idx

		while idx != end_idx + 1:
			tags[start_idx] = tag
			idx += 1

	return tags

def load_examples_file(file_name):
	examples = []

	with open(file_name, "r") as file:
		lines = file.readlines()

	sent = ""
	splitted_tags_idxs = defaultdict(list)

	for line in tqdm(lines, desc="Processing " + file_name, leave=False):
		line = line.replace("\n", "").replace("\r", "")

		# Is it a sentence or a list of arguments?
		if line.startswith("#"):
			if sent != "":
				examples.append((sent, splitted_tags_idxs))

			sent = line[2:]
			splitted_tags_idxs = defaultdict(list)
		elif line.startswith("+") or line.startswith("-"):
			tags = text_to_tags(sent, line[2:])

			splitted_tags_idxs[line[0]] += [encode_tags(tags)]

	examples.append((sent, splitted_tags_idxs))

	return examples



def main(action, train_filename, valid_filename):
	use_cuda = torch.cuda.is_available()
	#print(use_cuda)
	device = torch.device("cuda" if use_cuda else "cpu")
	print(device)

	if MODEL_NUM == 1:
		model = tagging_model(len(tags_dict.keys()))
	else: #if MODEL_NUM == 2:
		model = scoring_model(len(tags_dict.keys()))

	net = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).to(device)
	# There are only two possible actions- train and test
	if action == "-train":
		train_dataset = load_examples_file(train_filename)
		valid_dataset = load_examples_file(valid_filename)

		print("Train set size: ", len(train_dataset))
		print("Validation set size: ", len(valid_dataset))

		#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_factor)
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)

		train_losses = []
		train_accs = []
		valid_losses = []
		valid_accs = []

		for epoch in range(1, num_of_epochs + 1):
			# Training the model for an epoch
			train(net, device, train_dataset, optimizer, epoch)
			if not show_progress_bar: print("")

			# Validation on train and development sets
			train_loss, train_acc = valid(net, device, train_dataset, "Train", epoch)
			train_losses.append(train_loss)
			train_accs.append(train_acc)

			valid_loss, valid_acc = valid(model, device, valid_dataset, "Validation", epoch)
			valid_losses.append(valid_loss)
			valid_accs.append(valid_acc)

			print('Epoch ' + str(epoch) + ' - train loss = {:.4f}, train ACC = {:.3f}, valid loss = {:.4f}, valid ACC = {:.3f}'.format(train_loss, train_acc, valid_loss, valid_acc))
			if not show_progress_bar: print("")

			# Saving the trained models
			torch.save(net.state_dict(), trained_model_filename)

			# Saving the losses and the acc scores over training
			np.savetxt(CreateData.LEARNING_FILES_LOCATION + model_name + "_train_losses", np.array(train_losses))
			np.savetxt(CreateData.LEARNING_FILES_LOCATION + model_name + "_valid_losses", np.array(valid_losses))
			np.savetxt(CreateData.LEARNING_FILES_LOCATION + model_name + "_train_accs", np.array(train_accs))
			np.savetxt(CreateData.LEARNING_FILES_LOCATION + model_name + "_valid_accs", np.array(valid_accs))

		# Getting test results of the trained model
		#test(model, device, test_loader)

		# Showing the losses and acc graphs, over the number of epochs
		show_graph(train_accs, valid_accs, "Epoch Number", "AVG ACC")
		show_graph(train_losses, valid_losses, "Epoch Number", "CTC Loss")
	elif action == "-test":
		# Loading the trained model
		model.load_state_dict(torch.load(trained_model_filename))

		# Getting test results according the trained model
		#test(model, device, test_loader)



def show_graph(x_train_list, x_valid_list, x_label, y_label):
	"""
	Shows the graph of the given set (x, y)
	:param x_train_list: list of data from train
	:param x_valid_list: list of data from validation
	:param x_label: the x axis name
	:param y_label: the y axis name
	:return: None
	"""
	"""

	if type(x_train_list) != list:
		x_train_list = [x_train_list]

	if type(x_valid_list) != list:
		x_valid_list = [x_valid_list]

	plt.plot(range(len(x_train_list)), x_train_list, label="Train (" + str(round(x_train_list[-1], 3)) + ")")
	plt.plot(range(len(x_valid_list)), x_valid_list, label="Validation (" + str(round(x_valid_list[-1], 3)) + ")")
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.legend(loc='upper right')
	plt.show()
	"""

if __name__ == '__main__':
	# command line arguments:
	#	-train train_filename dev_filename
	#	-test train_filename dev_filename
	import sys

	print(tags_dict)

	if len(sys.argv) == 4:
		main(sys.argv[1], sys.argv[2], sys.argv[3])
	elif sys.argv[1] == "-graph":
		# Loading losses and accs files
		train_losses = np.loadtxt(CreateData.LEARNING_FILES_LOCATION + model_name + "_train_losses").tolist()
		valid_losses = np.loadtxt(CreateData.LEARNING_FILES_LOCATION + model_name + "_valid_losses").tolist()
		train_accs = np.loadtxt(CreateData.LEARNING_FILES_LOCATION + model_name + "_train_accs").tolist()
		valid_accs = np.loadtxt(CreateData.LEARNING_FILES_LOCATION + model_name + "_valid_accs").tolist()

		# Showing the losses and accs graphs
		show_graph(train_accs, valid_accs, "Epoch Number", "AVG ACC")
		show_graph(train_losses, valid_losses, "Epoch Number", "CTC Loss")