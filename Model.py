import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch_transformers import *

MODEL_NUM = 2
use_cuda = torch.cuda.is_available()

# Hyper Parameters
hyper_params = {
	"num_of_epochs":10,
	"optimizer":"adam",
	"learning_rate":0.001,
	"momentum_factor":0.75,
	"batch_size_limit":512,
	"model_name":"tagging_model" if MODEL_NUM == 1 else "scoring_model",
	"loss_type":"hinge-one-better", # hidge-all-better # nll
	"ignore_none_preds":True,
	"test_limit":5000,
	"device":torch.device("cuda" if use_cuda else "cpu"),
	"seed":5,
	"max_sent_size":35
}

class tagging_model(nn.Module):

	def __init__(self, tagset_size):
		super(tagging_model, self).__init__()

		# BERT Model
		self.bert = BertModel.from_pretrained('bert-base-uncased')
		for p in self.bert.embeddings.parameters():
			p.requires_grad = True

		# LSTM
		self.lstm = nn.LSTM(input_size=768, hidden_size=300 ,num_layers=2, batch_first=True, bidirectional=True, dropout=0.25)

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

		# LSTM
		packed_out = torch.nn.utils.rnn.pack_padded_sequence(bert_output[0], sents_lengths, batch_first=True)
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

		if len(self.batch_examples) + len(next_examples) <= hyper_params["batch_size_limit"]:
			self.batch_examples += next_examples
		elif len(next_examples) < hyper_params["batch_size_limit"]:
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

		padded_batch_tags = torch.Tensor(padded_batch_inputs[1]).long().to(hyper_params["device"])
		return torch.nn.functional.nll_loss(outputs, padded_batch_tags)

	def get_acc(self, batch_examples, outputs):
		"""
		Returns the accuracy of the given batch of examples, according to their model's outputs
		:param batch_examples: a list of batched examples
		:param outputs: the outputs of the model to those batch examples
		:return: accuracy score for the given batch
		"""

		batch_acc = 0

		preds = outputs.argmax(dim=1, keepdim=True).squeeze(dim=1)  # get the index of the max log-probability
		preds = preds.cpu().numpy()

		# Counts the right tagging for each example in the batch
		for i in range(len(preds)):
			count_right = 0
			count_wrong = 0

			for j in range(len(batch_examples[i][2])):
				if preds[i][j] == batch_examples[i][2][j]:
					# Ignoring NONE predicitons, if needed
					if preds[i][j] != 0 or not hyper_params["ignore_none_preds"]:
						count_right += 1
				else:
					count_wrong += 1

			batch_acc += count_right / (count_right + count_wrong)

		batch_acc /= len(batch_examples)

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

		if len(self.batch_examples) + len(next_examples) <= hyper_params["batch_size_limit"]:
			self.batch_examples += next_examples
		elif len(next_examples) < hyper_params["batch_size_limit"]:
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

		output_by_sents = []
		curr_outputs_for_sent = []

		for ex_idx in range(len(self.batch_examples)):
			sent_idx, splitted_sent, tags_idx, sent_length, is_right_tags = self.batch_examples[ex_idx]

			if last_sent_idx == -1:
				last_sent_idx = sent_idx

			# Is it a new sentence, or the last one?
			if sent_idx != last_sent_idx:
				# Calculating the loss of the last example
				if type(other_best_score) == torch.Tensor and type(right_score) == torch.Tensor:
					loss = torch.Tensor.max(torch.Tensor([0]).to(hyper_params["device"]),
											torch.Tensor([1]).to(hyper_params["device"]) - (right_score - other_best_score)).to(hyper_params["device"])
					batch_losses.append(loss)

				# Saving the relevant outputs (for ACC scoring)
				output_by_sents.append((curr_outputs_for_sent, right_score))

				last_sent_idx = sent_idx
				right_score = None
				other_best_score = None
				curr_outputs_for_sent = []

			curr_outputs_for_sent.append(outputs[ex_idx])

			if is_right_tags:
				# Saving the score of the right prediction
				if type(right_score) != torch.Tensor or \
						(hyper_params["loss_type"] == "hinge-all-better" and
						 torch.Tensor.equal(torch.Tensor.max(torch.Tensor([0]).to(hyper_params["device"]),
															 outputs[ex_idx] - right_score),
										    torch.Tensor([0]).to(hyper_params["device"]))) or \
						(hyper_params["loss_type"] == "hinge-one-better" and
						 not torch.Tensor.equal(torch.Tensor.max(torch.Tensor([0]).to(hyper_params["device"]),
																 outputs[ex_idx] - right_score),
												torch.Tensor([0]).to(hyper_params["device"]))):
					right_score = outputs[ex_idx]
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
		output_by_sents.append((curr_outputs_for_sent, right_score))

		self.batch_outputs = output_by_sents
		loss = sum(batch_losses) / len(batch_losses)

		return loss

	def get_acc(self, batch_examples, outputs):
		"""
		Returns the accuracy of the given batch of examples, according to their model's outputs
		:param batch_examples: a list of batched examples
		:param outputs: the outputs of the model to those batch examples
		:return: accuracy score for the given batch
		"""

		count_right = 0

		# Counts the examples in the batch with the right predicted arguments
		for (curr_outputs_for_sent, right_score) in outputs:
			if max(curr_outputs_for_sent) == right_score:
				count_right += 1

		batch_acc = count_right / len(outputs)

		return batch_acc

	def reset_model(self):
		self.batch_examples = []
		self.next_examples = []
		self.batch_outputs = None