import os
import operator
import torch
import torch.optim as optim
import numpy as np
import pickle
import random
from tqdm import tqdm
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

import DictsAndTables
import CreateData
from Model import tagging_model, scoring_model
from NomlexExtractor import load_txt_file
from CreateData import create_data
from NominalPatterns import clean_argument
from Model import MODEL_NUM, hyper_params

tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertTokenizer', 'bert-base-cased', do_basic_tokenize=False, do_lower_case=False)

# Constants
NUM_OF_EXAMPLES_TILL_TEST = 10000

comment = " ".join(sorted([param + "=" + str(value) for param, value in hyper_params.items()]))
tb_writer = SummaryWriter(comment=comment)
trained_model_filename = CreateData.LEARNING_FILES_LOCATION + "/trained_model " + comment
tags_dict_filename = CreateData.LEARNING_FILES_LOCATION + "tags_dict " + comment
print(comment)

tags_dict = {}
backward_tags_dict = {}
testing_idx = 1
model = None
constant_testing_indexes = []



def encode_tags(tags):
	"""
	Encoding the tags
	:param tags: a list of tags
	:return: a list of numbers (the encoded tags, based on the tags_dict dictionary)
	"""
	global tags_dict

	tags_idxs = []

	for i in range(len(tags)):
		# Updating the tags dictionary while encoding
		if tags[i] not in tags_dict.keys():
			new_index = len(tags_dict.keys())
			tags_dict[tags[i]] = new_index

		tags_idxs.append(tags_dict[tags[i]])

	return tags_idxs

def decode_tags(tags_idxs):
	"""
	Decoding the encoded tags
	:param tags_idxs: a list of encoded tags
	:return: a list of strings (the tags names, based on the inverted tags_dict dictionary)
	"""

	tags = []

	for i in range(len(tags_idxs)):
		tags.append(backward_tags_dict[tags_idxs[i]])

	return tags



def padding_by_batch(batch_examples):
	"""
	The forward process of the model, including aggregating all the losses
	:param batch_examples: a list of examples
	:return: the outputs of the model on the examples, and a list of losses based on those examples
	"""

	# Getting the length of the longest sentence in the batch
	max_length = max(batch_examples, key=operator.itemgetter(3))[3]

	# Sorting by sentence length first, then by sentence index and finally by right or wrong (right are first)
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
	padded_batch_indexed_tokens = torch.tensor(padded_batch_indexed_tokens).to(hyper_params["device"])
	padded_batch_tags = torch.tensor(padded_batch_tags).to(hyper_params["device"])
	sents_lengths = torch.tensor(sents_lengths).to(hyper_params["device"])

	return padded_batch_indexed_tokens, padded_batch_tags, sents_lengths

def train(net, train_dataset, valid_dataset, optimizer, epoch):
	"""
	Trains the model for a single epoch
	:param net: the trained model as dataparallel
	:param train_dataset: the training data
	:param valid_dataset: the validation data
	:param optimizer: the optimizer that will help training the model
	:param epoch: the number of the current epoch
	:return: None
	"""
	global testing_idx

	net.train()
	model.reset_model()

	random_indexes = np.arange(len(train_dataset))
	np.random.shuffle(random_indexes)

	sent_count = 0

	default_desc = 'Epoch ' + str(epoch) + ' [Training]'
	iterable = tqdm(random_indexes, desc=default_desc, leave=False)

	for sent_idx in iterable:
		sent, splitted_tags_idxs = train_dataset[sent_idx]
		sent_count += 1 # Counting the sentences (for progress printing)

		# Aggregating the batch examples
		is_full_batch = model.aggregate_batch(sent, splitted_tags_idxs, sent_idx)

		# Forward and backward for each batch separately
		if is_full_batch:
			optimizer.zero_grad()
			outputs = None

			# Forward
			padded_batch_inputs = padding_by_batch(model.batch_examples)
			try:
				outputs = net(padded_batch_inputs)
			except RuntimeError:
				with open("errors", "a") as errors_file:
					errors_file.write(str(padded_batch_inputs[0].shape) + "\n")
					errors_file.write(str(padded_batch_inputs[1].shape) + "\n")
					errors_file.write(str(padded_batch_inputs[2].shape) + "\n")
					errors_file.write(str(padded_batch_inputs[3]) + "\n")
					errors_file.write(str(padded_batch_inputs[4]) + "\n")
					errors_file.write(str(padded_batch_inputs[5]) + "\n")
					errors_file.write(str(model.batch_examples) + "\n")
					errors_file.write(str(sent_count) + "\n")
					errors_file.write(str(sent_idx) + "\n\n")

			if type(outputs) == torch.Tensor:
				model.batch_outputs = outputs

				# Backward
				loss = model.get_loss(outputs, padded_batch_inputs)
				loss.backward()
				optimizer.step()

				iterable.set_description(default_desc + ' - Loss = {:.6f}'.format(loss.item()))
				iterable.refresh()

			model.batch_examples = model.next_examples

		if sent_count % NUM_OF_EXAMPLES_TILL_TEST == 0:
			validate_model(net, train_dataset, valid_dataset, epoch=epoch)
			testing_idx += 1
			net.train()

	model.reset_model()

def valid(net, test_dataset, dataset_name, epoch, testing_count, test_on_random=False):
	"""
	Checks the prediction of the model over the validaiton data
	:param net: the trained model as dataparallel
	:param test_dataset: the validation data
	:param dataset_name: the name of the dataset (for printing)
	:param epoch: the number of the current epoch
	:param testing_count: the number of the current testing (in train, for printing)
	:param test_on_random: determine whether we will test on random indexes
	:return: loss, acc score
	"""

	net.eval()
	model.reset_model()

	test_loss = 0
	test_acc = 0

	if test_on_random:
		random_indexes = np.arange(len(test_dataset))
		np.random.shuffle(random_indexes)
		indexes = random_indexes[:hyper_params["test_limit"]]
	else:
		indexes = constant_testing_indexes.copy()

	sent_count = 0
	num_of_batches = 0

	default_desc = 'Epoch ' + str(epoch) + ' [' + dataset_name + ' test ' + str(testing_count) + ']'
	iterable = tqdm(indexes, desc=default_desc, leave=False)

	with torch.no_grad():
		for sent_idx in iterable:
			sent, splitted_tags_idxs = test_dataset[sent_idx]
			sent_count += 1  # Counting the sentences (for progress printing)

			# Aggregating the batch examples
			is_full_batch = model.aggregate_batch(sent, splitted_tags_idxs, sent_idx)

			# Forward and backward for each batch separately
			if is_full_batch:
				num_of_batches += 1
				outputs = None

				# Forward
				padded_batch_inputs = padding_by_batch(model.batch_examples)
				try:
					outputs = net(padded_batch_inputs)
				except RuntimeError:
					with open("errors", "a") as errors_file:
						errors_file.write(str(padded_batch_inputs[0].shape) + "\n")
						errors_file.write(str(padded_batch_inputs[1].shape) + "\n")
						errors_file.write(str(padded_batch_inputs[2].shape) + "\n")
						errors_file.write(str(padded_batch_inputs[3]) + "\n")
						errors_file.write(str(padded_batch_inputs[4]) + "\n")
						errors_file.write(str(padded_batch_inputs[5]) + "\n")
						errors_file.write(str(model.batch_examples) + "\n")
						errors_file.write(str(sent_count) + "\n")
						errors_file.write(str(sent_idx) + "\n\n")
						errors_file.write(str(model.batch_examples))

				if type(outputs) == torch.Tensor:
					model.batch_outputs = outputs

					# Calculate Loss
					loss = model.get_loss(outputs, padded_batch_inputs)
					test_loss += loss.item()

					# Calculate ACC
					acc = model.get_acc(model.batch_examples, model.batch_outputs)
					test_acc += acc

					iterable.set_description(default_desc + ' - loss = {:.6f}, ACC = {:.3f}'.format(loss.item(), acc))
					iterable.refresh()

				model.batch_examples = model.next_examples

	test_loss /= num_of_batches
	test_acc /= num_of_batches

	model.reset_model()

	return test_loss, test_acc

def test(model, test_dataset):
	"""
	Predicts over the test set, and returns the predictions
	:param model: the trained model
	:param test_dataset: the test dataset
	:return: a dictionary of predictions ({sentence: {(nom, index}: arguments})
	"""

	model.eval()
	predictions = {}
	model.reset_model()

	DictsAndTables.should_clean = True

	with torch.no_grad():
		iterable = tqdm(range(len(test_dataset)), desc="Testing", leave=False)
		for sent_idx in iterable:
			if MODEL_NUM == 2:
				sent, splitted_tags = test_dataset[sent_idx]
			else: # if MODEL_NUM == 1:
				if type(test_dataset[sent_idx]) == tuple:
					sent, _ = test_dataset[sent_idx]
				else:
					sent = test_dataset[sent_idx]

			splitted_sent = sent.split(" ")

			indexed_tokens = tokenizer.convert_tokens_to_ids(splitted_sent)
			sent_length = len(splitted_sent)

			if MODEL_NUM == 1:
				batch_indexed_tokens = torch.tensor([indexed_tokens]).to(hyper_params["device"])
				sents_lengths = torch.tensor([sent_length]).to(hyper_params["device"])

				# Calling the model to get the output
				output = model(batch_indexed_tokens, sents_lengths).permute(0, 2, 1)

				# Get the predicted tags for the sentence
				encoded_pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
				pred_tags = decode_tags(encoded_pred)

				if type(predictions.get(sent, -1)) != dict:
					predictions[sent] = {}

				# Find the nominalization in the founded tag list
				nom_idx = -1
				for i in range(len(pred_tags)):
					if pred_tags[i] == "NOM":
						nom_idx = i

				if nom_idx != -1:
					predictions[sent][(splitted_sent[nom_idx], nom_idx)] = tags_to_arguments(pred_tags, splitted_sent)

			else: # if MODEL_NUM == 2:
				batch_tags = []
				sents_lengths = []
				batch_indexed_tokens = []

				for tags in splitted_tags['-']:
					batch_tags.append(encode_tags(tags))
					batch_indexed_tokens.append(indexed_tokens)
					sents_lengths.append(sent_length)

				batch_tags = torch.tensor(batch_tags).to(hyper_params["device"])
				sents_lengths = torch.tensor(sents_lengths).to(hyper_params["device"])
				batch_indexed_tokens = torch.tensor(batch_indexed_tokens).to(hyper_params["device"])

				# Calling the model to get the output
				outputs = model(batch_indexed_tokens, batch_tags, sents_lengths)

				# Get the predicted tags for the sentence
				pred_idx = outputs.argmax(keepdim=True)
				pred_tags = splitted_tags['-'][pred_idx]

				if type(predictions.get(sent, -1)) != dict:
					predictions[sent] = {}

				# Find the nominalization in the founded tag list
				nom_idx = -1
				for i in range(len(pred_tags)):
					if "NOM" in pred_tags[i]:
						nom_idx = i

				if nom_idx != -1:
					predictions[sent][(splitted_sent[nom_idx], nom_idx)] = tags_to_arguments(pred_tags, splitted_sent)

	model.reset_model()

	return predictions



def tags_to_arguments(tags, splitted_sent):
	"""
	Translates the given list of tags that is suitable for each word in the given sentence, into dictionary of arguments
	:param tags: a list of tags
	:param splitted_sent: a splitted sentence (list of words)
	:return: a dictionary of arguments
	"""

	arguments = {}
	last_tag = ("NONE", "")

	for i in range(len(tags)):
		if last_tag[0] == tags[i]:
			if tags[i] != "NONE":
				last_tag = (last_tag[0], last_tag[1] + " " + splitted_sent[i])
		else:
			if last_tag[0] != "NONE":
				arguments[last_tag[0]] = clean_argument(last_tag[1])
				if tags[i] != "NONE":
					last_tag = (tags[i], splitted_sent[i])
			else:
				last_tag = (tags[i], splitted_sent[i])

	if last_tag[0] != "NONE":
		arguments[last_tag[0]] = clean_argument(last_tag[1])

	return arguments

def text_to_tags(sentence, arguments_as_text):
	"""
	Translating the text of arguments from a specific pattern, into a list of tags (a tag for each word in the sentence)
	:param sentence: the sentence (str)
	:param arguments_as_text: a text of arguments from a specific pattern
	:return: a list of tags
	"""

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
	"""
	Loads a file with examples.
	The looks like- sentence (starts with #), patterns (starts with + or -) each per line, sentence again, and so on
	:param file_name: the location of the file
	:return: The examples loaded from the file
	"""

	examples = []

	with open(file_name, "r") as file:
		lines = file.readlines()

	sent = ""
	splitted_tags_idxs = defaultdict(list)
	legitimate_sent = False

	for line in tqdm(lines, desc="Processing " + file_name, leave=False):
		line = line.replace("\n", "").replace("\r", "")

		# Is it a sentence or a list of arguments?
		if line.startswith("#"):
			if sent != "" and legitimate_sent:
				examples.append((sent, splitted_tags_idxs))

			sent = line[2:]
			splitted_tags_idxs = defaultdict(list)
			legitimate_sent = len(sent.split(" ")) <= hyper_params["max_sent_size"]

		elif legitimate_sent and (line.startswith("+") or line.startswith("-")):
			tags = text_to_tags(sent, line[2:])
			splitted_tags_idxs[line[0]] += [encode_tags(tags)]

	# Adding also the last examples
	if legitimate_sent:
		examples.append((sent, splitted_tags_idxs))

	return examples

def load_train_and_dev(train_filename, dev_filename):
	"""
	Loads the training and development datasets from the files with the given name
	:param train_filename: the name of the training set's file
	:param dev_filename: the name of the development set's file
	:return: train dataset and development dataset
	"""
	global constant_testing_indexes

	train_dataset = load_examples_file(train_filename)
	valid_dataset = load_examples_file(dev_filename)
	print("Train set size: ", len(train_dataset))
	print("Validation set size: ", len(valid_dataset))

	random.seed(a=hyper_params["seed"])
	constant_testing_indexes = random.sample(range(len(valid_dataset)), min(hyper_params["test_limit"], len(valid_dataset)))
	random.seed(a=None)

	return train_dataset, valid_dataset

def load_trained_model(model_filename, tags_dict_filename, should_retrain=True):
	"""
	Loads a trained model according to the current hyper parameters
	:param model_filename: the name of the model's file
	:param tags_dict_filename: the name of the tags dicitonary's file
	:param should_retrain: determines whether the loaded model should retrain, or train from the begining
	:return: the trained model, and its network version as data parrallel (for multi-GPU training)
	"""

	global tags_dict, backward_tags_dict, model

	# Loading the dictionary of tags
	if os.path.isfile(tags_dict_filename):
		with open(tags_dict_filename, "rb") as tags_dict_file:
			tags_dict = pickle.load(tags_dict_file)
	else:
		# Saving the dictionary of tags
		with open(tags_dict_filename, "wb") as tags_dict_file:
			pickle.dump(tags_dict, tags_dict_file)

	backward_tags_dict = dict([(v, k) for k, v in tags_dict.items()])

	# Creating the model
	if MODEL_NUM == 1:
		model = tagging_model(len(tags_dict.keys())).to(hyper_params["device"])
		print("MODEL: tagging model")
	else:  # if MODEL_NUM == 2:
		model = scoring_model(len(tags_dict.keys())).to(hyper_params["device"])
		print("MODEL: scoring model")

	# Loading the trained model, if it exists and if it is needed
	if os.path.isfile(model_filename):
		if should_retrain:
			model.load_state_dict(torch.load(model_filename))
		else:
			if input("A trained model with the given hyper parameters was found. Do you sure you want to replace it? (y/n): ") != 'y':
				exit()

	else:
		if should_retrain:
			print("A trained model with the given hyper parameters wasn't found!")
			exit()

	net = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).to(hyper_params["device"])

	return model, net



def train_model(net, train_dataset, valid_dataset):
	"""
	Trains the given model over the train and development datasets
	:param net: the trained model as dataparallel
	:param train_dataset: the train dataset
	:param valid_dataset: the development dataset
	:return: None
	"""
	global model, tags_dict, backward_tags_dict, testing_idx

	# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_factor)
	if hyper_params["optimizer"] == "adam":
		optimizer = optim.Adam(model.parameters(), lr=hyper_params["learning_rate"])
	else: #if hyper_params["optimizer"] == "SGD":
		optimizer = optim.SGD(model.parameters(), lr=hyper_params["learning_rate"], momentum=hyper_params["momentum_factor"])

	for epoch in tqdm(range(1, hyper_params["num_of_epochs"] + 1), desc="Learning", leave=False):
		# Training the model for an epoch
		train(net, train_dataset, valid_dataset, optimizer, epoch)

		# Validating the model on the training an validation sets
		validate_model(net, train_dataset, valid_dataset, epoch=epoch)

		testing_idx = 1

def validate_model(net, train_dataset, valid_dataset, epoch=None):
	"""
	Validates the given model over the train and development datasets
	:param net: the trained model as dataparallel
	:param train_dataset: the train dataset
	:param valid_dataset: the development dataset
	:param epoch: the epoch number, optional (useful for validation during training)
	:return: None
	"""
	global testing_idx

	# Testing on train set
	rand_train_loss, rand_train_acc = valid(net, train_dataset, "Random train", epoch, testing_idx, test_on_random=True)
	tb_writer.add_scalar('Loss/rand_train', rand_train_loss, testing_idx)
	tb_writer.add_scalar('Accuracy/rand_train', rand_train_acc, testing_idx)
	train_loss, train_acc = valid(net, train_dataset, "Train", epoch, testing_idx, test_on_random=False)
	tb_writer.add_scalar('Loss/train', train_loss, testing_idx)
	tb_writer.add_scalar('Accuracy/train', train_acc, testing_idx)

	# Testing on validation set
	rand_valid_loss, rand_valid_acc = valid(net, valid_dataset, "Random validation", epoch, testing_idx, test_on_random=True)
	tb_writer.add_scalar('Loss/rand_valid', rand_valid_loss, testing_idx)
	tb_writer.add_scalar('Accuracy/rand_valid', rand_valid_acc, testing_idx)
	valid_loss, valid_acc = valid(net, valid_dataset, "Validation", epoch, testing_idx, test_on_random=False)
	tb_writer.add_scalar('Loss/valid', valid_loss, testing_idx)
	tb_writer.add_scalar('Accuracy/valid', valid_acc, testing_idx)

	curr_results_str = ''

	if not epoch:
		curr_results_str += 'Epoch ' + str(epoch) + ' Testing ' + str(testing_idx) + ' - '

	curr_results_str = 'train loss = {:.4f} vs rand {:.4f}, train ACC = {:.3f} vs rand {:.3f}, valid loss = {:.4f} vs rand {:.4f}, valid ACC = {:.3f} vs rand {:.3f}'.format(
						train_loss, rand_train_loss, train_acc, rand_train_acc, valid_loss, rand_train_acc, valid_acc, rand_valid_acc)
	tqdm.write(curr_results_str)

	# Saving the trained model
	torch.save(model.state_dict(), trained_model_filename)

	for name, weight in net.named_parameters():
		tb_writer.add_histogram(name, weight, testing_idx)
		tb_writer.add_histogram(f'{name}.grad', weight.grad, testing_idx)

	testing_idx += 1

def test_model(model, nomlex_filename, test_data):
	"""
	Tests the given model over the given testing data
	:param model: the trained model
	:param nomlex_filename: a location of a json file with the entries of NOMLEX lexicon
	:param test_data: the testing data, a list of sentences
	:return: The model's predictions ({sent: {nom: arguments}}
	"""

	if MODEL_NUM == 2:
		test_dataset = create_data(nomlex_filename, test_data, write_to_files=False, ignore_right=True, use_catvar=True)
	else:  # if MODEL_NUM == 1:
		test_dataset = test_data

	# Getting test results according the trained model
	results = test(model, test_dataset)

	for sent in results.keys():
		print(sent)

		for nom, arguments in results[sent].items():
			print("--" + str(nom) + ": " + str(arguments))

		print("")

	return results



if __name__ == '__main__':
	# command line arguments:
	#	-train train_filename dev_filename
	#	-retrain train_filename dev_filename
	#	-valid train_filename dev_filename
	#	-test nomlex_filename sentence
	#	-test nomlex_filename test_filename
	#
	# Examples:
	# 	python Learning.py -train learning/train_x00.parsed learning/valid_x00.parsed
	# 	python Learning.py -test NOMLEX_Data/NOMLEX-plus-only-nom.json "The appointment of Alice by Apple"
	import sys

	if len(sys.argv) == 4:
		action = sys.argv[1]

		if action == "-train":
			train_dataset, valid_dataset = load_train_and_dev(sys.argv[2], sys.argv[3])
			model, net = load_trained_model(trained_model_filename, tags_dict_filename, should_retrain=False)
			train_model(net, train_dataset, valid_dataset)

		elif action == "-retrain":
			train_dataset, valid_dataset = load_train_and_dev(sys.argv[2], sys.argv[3])
			model, net = load_trained_model(trained_model_filename, tags_dict_filename, should_retrain=True)
			train_model(net, train_dataset, valid_dataset)

		elif action == "-valid":
			train_dataset, valid_dataset = load_train_and_dev(sys.argv[2], sys.argv[3])
			model, net = load_trained_model(trained_model_filename, tags_dict_filename, should_retrain=True)
			validate_model(net, train_dataset, valid_dataset)

		elif action == "-test":
			nomlex_filename = sys.argv[2]
			model, net = load_trained_model(trained_model_filename, tags_dict_filename, should_retrain=True)

			# Loading test data
			if os.path.isfile(sys.argv[3]):
				test_data = load_txt_file(sys.argv[3])
			else:
				test_data = [sys.argv[3]]  # This is actually a single example sentence
			test_model(model, sys.argv[2], test_data)

	tb_writer.close()