import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Model that classifies the argument type based on a pre-trained encoder, like BERT
class ArgumentsModel(nn.Module):
	def __init__(self, n_labels, bert_wrapper):
		super(ArgumentsModel, self).__init__()

		self.pretrained_encoder = bert_wrapper.bert
		bert_dim = bert_wrapper.dim # 768

		self.fc1 = torch.nn.Linear(bert_dim, int(bert_dim / 2))
		self.dropout = torch.nn.Dropout(p=0.5)
		self.fc2 = torch.nn.Linear(int(bert_dim / 2), n_labels)

	def forward(self, token_ids, token_mask, start_argument_index, end_argument_index, predicate_index, excluded_tagset_ids=None):
		# truncate to longest sequence length in batch (usually much smaller than 512) to save GPU RAM
		max_length = (token_mask != 0).max(0)[0].nonzero()[-1].item()
		if max_length < token_ids.shape[1]:
			token_ids = token_ids[:, :max_length]
			token_mask = token_mask[:, :max_length]

		bert_last_layer = self.pretrained_encoder(token_ids, token_mask)[0]
		# bert_last_layer = self.bert(input_token_ids)[0]

		# mlp_in = torch.cat([bert_last_layer[input_argument_index], bert_last_layer[input_predicate_index]], dim=1)
		batch_range = torch.arange(bert_last_layer.size(0))
		mlp_in = bert_last_layer[batch_range, start_argument_index] + bert_last_layer[batch_range, end_argument_index] + bert_last_layer[batch_range, predicate_index]

		# output/classification layer: input bert states and get log probabilities for cross entropy loss
		fc1_out = self.fc1(self.dropout(mlp_in))
		fc2_out = self.fc2(F.relu(fc1_out))

		if excluded_tagset_ids is not None:
			fc2_out[:, excluded_tagset_ids] = -np.inf

		logits = F.log_softmax(fc2_out, dim=1)

		return logits