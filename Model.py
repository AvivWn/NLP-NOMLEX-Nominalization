import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch_transformers import *
from operator import itemgetter

import Learning
from Learning import decode_tags, tags_to_arguments

MODEL_NUM = 2
use_cuda = torch.cuda.is_available()

# Hyper Parameters
hyper_params = {
	"epochs":10,
	"optimizer":"adam", # "SGD"
	"lr":0.001,
	"batch_limit":256,
	"model":"tagging_model" if MODEL_NUM == 1 else "scoring_model",
	"loss":"hinge-all-better", # "hinge-one-better", # "hinge-all-better" # "nll"
	"device":torch.device("cuda" if use_cuda else "cpu"),
	"max_sent":35,
	#"seed": 5,
	#"test_limit": 10000,
}

if hyper_params["optimizer"] == "SGD":
	hyper_params.update({"momentum":0.75})

if MODEL_NUM == 1:
	hyper_params.update({"ignore_none":True})

class tagging_model(nn.Module):

	def __init__(self, tagset_size):
		super(tagging_model, self).__init__()

		# zero-one embeddings (1 is NOM and zero is all the others)
		self.tag_embedding = nn.Embedding(2, tagset_size)

		# BERT Model
		self.bert = BertModel.from_pretrained('bert-base-uncased')
		for p in self.bert.embeddings.parameters():
			p.requires_grad = True

		# LSTM
		self.lstm = nn.LSTM(input_size=768 + tagset_size, hidden_size=300 ,num_layers=2, batch_first=True, bidirectional=True, dropout=0.25)

		# Fully Connected Layers
		self.fc1 = nn.Linear(600, 600, 600)
		self.fc2 = nn.Linear(600, tagset_size, tagset_size)

		# Initiating the weights
		nn.init.xavier_normal_(self.fc1.weight)
		nn.init.xavier_normal_(self.fc2.weight)

		# For batching
		self.batch_examples = []
		self.next_examples = []
		self.batch_outputs = None

	def forward(self, padded_batch_inputs):
		"""
		The forward process of the model
		:param padded_batch_inputs: all the inputs needed for the model, for a single batch, after padding them
		:return: the model's output
		"""

		padded_batch_indexed_tokens, padded_batch_tags, sents_lengths = padded_batch_inputs

		# Moving to bert embeddings
		bert_output = self.bert(padded_batch_indexed_tokens)

		# Adding the conditions according to the given tags, conditioned the nominalization in the sentence
		conditions = (padded_batch_tags == int(Learning.tags_dict['NOM'])).float().long()
		conditioned_bert = torch.cat([self.tag_embedding(conditions), bert_output[0]], dim=2)

		# LSTM
		packed_out = torch.nn.utils.rnn.pack_padded_sequence(conditioned_bert, sents_lengths, batch_first=True)
		self.lstm.flatten_parameters()
		lstm_out = self.lstm(packed_out)[0]
		padded_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=padded_batch_indexed_tokens.shape[1])

		# Fully Connected Layers
		fc1_out = F.relu(self.fc1(padded_out))
		fc2_out = self.fc2(fc1_out)

		return F.log_softmax(fc2_out, dim=2).permute(0, 2, 1)

	def aggregate_batch(self, sent, splitted_tags_idxs, sent_idx):
		"""
		Aggregates the given data, into the current batch
		:param sent: a sentence (str)
		:param splitted_tags_idxs: a dictionary of list of tags, splitted to right (+) and wrong (-) tags, for the given sentence
		:param sent_idx: the index of the sentence (int)
		:return: a boolean flag that determines whether the batch is already full or not
		"""

		next_examples = []

		right_tags_list = splitted_tags_idxs["+"]

		if len(right_tags_list) == 0:
			return False

		splitted_sent = sent.split(" ")
		next_examples.append((sent_idx, splitted_sent, right_tags_list[0], len(splitted_sent), True))

		is_full_batch = False

		if len(self.batch_examples) + len(next_examples) <= hyper_params["batch_limit"]:
			self.batch_examples += next_examples
		elif len(next_examples) < hyper_params["batch_limit"]:
			self.next_examples = next_examples
			is_full_batch = True

		return is_full_batch

	def get_loss(self, outputs, padded_batch_inputs):
		"""
		Returns the loss of the model, according to the given outputs of the model and the given suitable inputs
		:param outputs: the output tensor of the model
		:param padded_batch_inputs: all the inputs needed for the model, for a single batch, after padding them
		:return: the loss of the model
		"""

		padded_batch_tags = padded_batch_inputs[1].long()
		return torch.nn.functional.nll_loss(outputs, padded_batch_tags)

	def get_acc(self, batch_examples, outputs, backward_tags_dict, model_errors_file=None):
		"""
		Returns the accuracy of the given batch of examples, according to their model's outputs
		:param batch_examples: a list of batched examples
		:param outputs: the outputs of the model to those batch examples
		:param backward_tags_dict: a backward tags dictionary ({tag_id: tag})
		:param model_errors_file: a file for writing the mistakes of the model, optional
		:return: accuracy score for the given batch
		"""

		batch_acc = 0

		preds = outputs.argmax(dim=1, keepdim=True).squeeze(dim=1)  # get the index of the max log-probability
		preds = preds.cpu().numpy()

		# Counts the right tagging for each example in the batch
		for i in range(len(preds)):
			count_right = 0
			count_wrong = 0
			pred_tags = list(preds[i])
			right_tags = batch_examples[i][2]

			for j in range(len(batch_examples[i][2])):
				if preds[i][j] == batch_examples[i][2][j]:
					# Ignoring NONE predicitons, if needed
					if preds[i][j] != 0 or not hyper_params["ignore_none"]:
						count_right += 1
				else:
					count_wrong += 1

			if pred_tags != right_tags and model_errors_file:
				splitted_sent = batch_examples[i][1]
				prediction_args = tags_to_arguments(decode_tags(pred_tags, backward_tags_dict), splitted_sent)
				right_args = tags_to_arguments(decode_tags(right_tags, backward_tags_dict), splitted_sent)
				model_errors_file.write(" ".join(splitted_sent) + "\nPrediction: " + str(prediction_args) + "\nRight: " + str(right_args) + "\n\n")
				model_errors_file.flush()

			batch_acc += count_right / (count_right + count_wrong)

		batch_acc /= len(batch_examples)

		return batch_acc

	def reset_model(self):
		self.batch_examples = []
		self.next_examples = []
		self.batch_outputs = None


class scoring_model(nn.Module):

	def __init__(self, tagset_size):
		super(scoring_model, self).__init__()

		# Tag Embedding
		self.tag_embedding = nn.Embedding(tagset_size, tagset_size)

		# BERT Model
		#self.tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertTokenizer', 'bert-base-uncased', do_basic_tokenize=False, do_lower_case=False)
		self.bert = BertModel.from_pretrained('bert-base-uncased')
		for p in self.bert.embeddings.parameters():
			p.requires_grad = True

		# LSTM
		self.lstm = nn.LSTM(input_size=768 + tagset_size, hidden_size=150 ,num_layers=2, batch_first=True, bidirectional=True, dropout=0.25)

		# Fully Connected Layers
		self.fc1 = nn.Linear(300, 100, 100)
		self.fc2 = nn.Linear(100, 1, 1)

		# Initiating the weights
		nn.init.xavier_normal_(self.fc1.weight)
		nn.init.xavier_normal_(self.fc2.weight)

		# For batching
		self.batch_examples = []
		self.next_examples = []
		self.batch_outputs = None

	def forward(self, padded_batch_inputs):
		"""
		The forward process of the model
		:param padded_batch_inputs: all the inputs needed for the model, for a single batch, after padding them
		:return: the model's output
		"""

		padded_batch_indexed_tokens, padded_batch_tags, sents_lengths = padded_batch_inputs

		# Moving to bert embeddings
		bert_output = self.bert(padded_batch_indexed_tokens)

		# Adding the conditions according to the given tags
		conditions = self.tag_embedding(padded_batch_tags)
		conditioned_bert = torch.cat([conditions, bert_output[0]], dim=2)

		# LSTM
		if self.lstm.bidirectional:
			num_directions = 2
		else:
			num_directions = 1

		packed_out = torch.nn.utils.rnn.pack_padded_sequence(conditioned_bert, sents_lengths, batch_first=True)
		self.lstm.flatten_parameters()
		lstm_out = self.lstm(packed_out)[1][0]

		# Rearranging lstm output (we want only the last hidden state of the last layer)
		lstm_out = lstm_out.view(self.lstm.num_layers, num_directions, lstm_out.shape[1], self.lstm.hidden_size)[-1]
		lstm_out = torch.cat([lstm_out[0], lstm_out[1]], dim=1)

		# Fully Connected Layers
		fc1_out = F.relu(self.fc1(lstm_out))
		fc2_out = self.fc2(fc1_out).squeeze(dim=0)

		return fc2_out

	def aggregate_batch(self, sent, splitted_tags_idxs, sent_idx):
		"""
		Aggregates the given data, into the current batch
		:param sent: a sentence (str)
		:param splitted_tags_idxs: a dictionary of list of tags, splitted to right (+) and wrong (-) tags, for the given sentence
		:param sent_idx: the index of the sentence (int)
		:return: a boolean flag that determines whether the batch is already full or not
		"""

		next_examples = []

		right_tags_idxs_list = splitted_tags_idxs["+"]
		wrong_tags_idxs_list = splitted_tags_idxs["-"]
		splitted_sent = sent.split(" ")

		if len(wrong_tags_idxs_list) != 0 and len(right_tags_idxs_list) != 0:
			for right_tags_idxs in right_tags_idxs_list:
				next_examples.append((sent_idx, splitted_sent, right_tags_idxs, len(splitted_sent), True))

			for wrong_tags_idxs in wrong_tags_idxs_list:
				next_examples.append((sent_idx, splitted_sent, wrong_tags_idxs, len(splitted_sent), False))

		is_full_batch = False

		if len(self.batch_examples) + len(next_examples) <= hyper_params["batch_limit"]:
			self.batch_examples += next_examples
		elif len(next_examples) < hyper_params["batch_limit"]:
			self.next_examples = next_examples
			is_full_batch = True

		return is_full_batch

	def get_loss(self, outputs, padded_batch_inputs):
		"""
		Returns the loss of the model, according to the given outputs of the model and the given suitable inputs
		:param outputs: the output tensor of the model
		:param padded_batch_inputs: all the inputs needed for the model, for a single batch, after padding them
		:return: the loss of the model
		"""

		batch_losses = []
		right_score = None
		other_best_score = None
		last_sent_idx = -1
		last_splitted_sent = []

		output_by_sents = []
		right_scores = []
		all_scores = []

		for ex_idx in range(len(self.batch_examples)):
			sent_idx, splitted_sent, tags_idx, sent_length, is_right_tags = self.batch_examples[ex_idx]

			if last_sent_idx == -1:
				last_sent_idx = sent_idx
				last_splitted_sent = splitted_sent

			# Is it a new sentence, or the last one?
			if sent_idx != last_sent_idx:
				# Calculating the loss of the last example
				if type(other_best_score) == torch.Tensor and type(right_score) == torch.Tensor:
					loss = torch.Tensor.max(torch.Tensor([0]).to(hyper_params["device"]),
											torch.Tensor([1]).to(hyper_params["device"]) - (right_score - other_best_score)).to(hyper_params["device"])
					batch_losses.append(loss)

				# Saving the relevant outputs (for ACC scoring)
				output_by_sents.append((last_splitted_sent, all_scores, right_scores))

				last_sent_idx = sent_idx
				last_splitted_sent = splitted_sent
				right_score = None
				other_best_score = None
				all_scores = []
				right_scores = []

			all_scores.append((tags_idx, outputs[ex_idx]))

			if is_right_tags:
				# Saving the score of the right prediction
				if type(right_score) != torch.Tensor or \
						(hyper_params["loss"] == "hinge-all-better" and
						 torch.Tensor.equal(torch.Tensor.max(torch.Tensor([0]).to(hyper_params["device"]),
															 outputs[ex_idx] - right_score),
										    torch.Tensor([0]).to(hyper_params["device"]))) or \
						(hyper_params["loss"] == "hinge-one-better" and
						 not torch.Tensor.equal(torch.Tensor.max(torch.Tensor([0]).to(hyper_params["device"]),
																 outputs[ex_idx] - right_score),
												torch.Tensor([0]).to(hyper_params["device"]))):
					right_score = outputs[ex_idx]

				right_scores.append((tags_idx, outputs[ex_idx]))
			else:
				# Saving the score of the other prediction (not right prediction), with the best score
				if type(other_best_score) != torch.Tensor or \
						not torch.Tensor.equal(torch.Tensor.max(torch.Tensor([0]).to(hyper_params["device"]),
																outputs[ex_idx] - other_best_score),
											   torch.Tensor([0]).to(hyper_params["device"])):
					other_best_score = outputs[ex_idx]

		# Calculating the loss of the last example
		if type(other_best_score) == torch.Tensor and type(right_score) == torch.Tensor:
			loss = torch.Tensor.max(torch.Tensor([0]).to(hyper_params["device"]),
									torch.Tensor([1]).to(hyper_params["device"]) - (
												right_score - other_best_score)).to(hyper_params["device"])
			batch_losses.append(loss)

		# Saving the relevant outputs (for ACC scoring)
		output_by_sents.append((last_splitted_sent, all_scores, right_scores))

		self.batch_outputs = output_by_sents
		loss = sum(batch_losses) / len(batch_losses)

		return loss

	def get_acc(self, batch_examples, outputs, backward_tags_dict, model_errors_file=None):
		"""
		Returns the accuracy of the given batch of examples, according to their model's outputs
		:param batch_examples: a list of batched examples
		:param outputs: the outputs of the model to those batch examples
		:param backward_tags_dict: a backward tags dictionary ({tag_id: tag})
		:param model_errors_file: a file for writing the mistakes of the model, optional
		:return: accuracy score for the given batch
		"""

		count_right = 0

		# Counts the examples in the batch with the right predicted arguments
		for (splitted_sent, all_scores, right_scores) in outputs:
			prediction = max(all_scores, key=itemgetter(1))
			if prediction in right_scores:
				count_right += 1
			else:
				if model_errors_file:
					prediction_args = tags_to_arguments(decode_tags(prediction[0], backward_tags_dict), splitted_sent)
					right_args = tags_to_arguments(decode_tags(right_scores[0][0], backward_tags_dict), splitted_sent)
					model_errors_file.write(" ".join(splitted_sent) + "\nPrediction: " + str(prediction_args) + "\nRight: " + str(right_args) + "\n\n")
					model_errors_file.flush()

		batch_acc = count_right / len(outputs)

		return batch_acc

	def reset_model(self):
		self.batch_examples = []
		self.next_examples = []
		self.batch_outputs = None