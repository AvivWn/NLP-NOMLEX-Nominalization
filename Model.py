import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch_transformers import *

class tagging_model(nn.Module):

	def __init__(self, tagset_size):
		super(tagging_model, self).__init__()

		self.bert = BertModel.from_pretrained('bert-base-uncased')

		#self.fc = nn.Linear(768, 100, 100)

		self.lstm = nn.LSTM(input_size=768, hidden_size=50 ,num_layers=2, batch_first=True, bidirectional=True, dropout=0.25)

		self.fc1 = nn.Linear(100, 100, 100)
		self.fc2 = nn.Linear(100, tagset_size, tagset_size)

		nn.init.xavier_normal_(self.fc1.weight)
		nn.init.xavier_normal_(self.fc2.weight)

	def forward(self, padded_batch_indexed_tokens, sents_lengths):
		# Moving to bert embeddings
		bert_output = self.bert(padded_batch_indexed_tokens)

		# RNN
		packed_out = torch.nn.utils.rnn.pack_padded_sequence(bert_output[0], sents_lengths, batch_first=True)
		lstm_out = self.lstm(packed_out)[0]
		padded_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=padded_batch_indexed_tokens.shape[1])

		# Fully Connected
		fc1_out = F.relu(self.fc1(padded_out))
		fc2_out = self.fc2(fc1_out)

		return F.log_softmax(fc2_out, dim=2)




class scoring_model(nn.Module):

	def __init__(self, tagset_size):
		super(scoring_model, self).__init__()

		self.tag_embedding = nn.Embedding(tagset_size, tagset_size)

		#self.tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertTokenizer', 'bert-base-uncased', do_basic_tokenize=False, do_lower_case=False)
		self.bert = BertModel.from_pretrained('bert-base-uncased')

		self.lstm = nn.LSTM(input_size=768 + tagset_size, hidden_size=150 ,num_layers=2, batch_first=True, bidirectional=True, dropout=0.25)

		self.fc1 = nn.Linear(300, 100, 100)
		self.fc2 = nn.Linear(100, 1, 1)

		#nn.init.xavier_normal_(self.lstm.all_weights)
		nn.init.xavier_normal_(self.fc1.weight)
		nn.init.xavier_normal_(self.fc2.weight)

	def forward(self, padded_batch_indexed_tokens, padded_batch_tags, sents_lengths):
		# Moving to bert embeddings
		bert_output = self.bert(padded_batch_indexed_tokens)

		# Adding the conditions according to the given tags
		conditions = self.tag_embedding(padded_batch_tags)
		conditioned_bert = torch.cat([conditions, bert_output[0]], dim=2)

		# RNN
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

		# Fully Connected
		fc1_out = F.relu(self.fc1(lstm_out))
		fc2_out = self.fc2(fc1_out).squeeze(dim=0).squeeze(dim=1)

		return fc2_out