import torch
import numpy as np
from pytorch_transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel

from arguments_extractor.utils import flatten

class PretrainedEncoder:
	MASK = '[MASK]'
	CLS = "[CLS]"
	SEP = "[SEP]"
	START_ARG = "<ARG>"
	END_ARG = "</ARG>"

	def __init__(self, model_name):
		super().__init__()
		self.model_name = model_name

		if model_name.startswith("bert"):
			# BERT
			do_lower_case = "uncased" in model_name
			self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=do_lower_case)
			self.bert = BertModel.from_pretrained(self.model_name)

		else:
			# RoBERTa
			self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
			self.bert = RobertaModel.from_pretrained(model_name)
			self.CLS = "<s>"
			self.SEP = "</s>"

		self.max_len = self.bert.embeddings.position_embeddings.weight.size(0)
		self.dim = self.bert.embeddings.position_embeddings.weight.size(1)

	def convert_tokens_to_ids(self, tokens):
		token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
		ids = torch.tensor([token_ids])
		assert ids.size(1) < self.max_len

		padded_ids = torch.zeros(1, self.max_len).to(ids)
		padded_ids[0, :ids.size(1)] = ids
		mask = torch.zeros(1, self.max_len).to(ids)
		mask[0, :ids.size(1)] = 1
		return padded_ids, mask

	def subword_tokenize(self, tokens, start_argument_index, end_argument_index, predicate_index, suitable_verb):
		subwords = list(map(self.tokenizer.tokenize, tokens))
		subword_lengths = list(map(len, subwords))

		subwords = [self.CLS] + flatten(subwords[:start_argument_index]) + \
				   [self.START_ARG] + flatten(subwords[start_argument_index:end_argument_index+1]) + [self.END_ARG] + \
				   flatten(subwords[end_argument_index+1:]) + [self.SEP] + list(self.tokenizer.tokenize(suitable_verb)) + [self.SEP]
					# flatten(subwords[end_argument_index+1:]) + [self.SEP] + subwords[predicate_index] + [self.SEP]

		subword_lengths = [0] + subword_lengths
		subword_lengths[end_argument_index + 1] += 1
		subword_lengths[start_argument_index] += 1
		token_start_idxs = 1 + np.cumsum(subword_lengths[:-1])
		return subwords, token_start_idxs

	def subword_tokenize_to_ids(self, tokens, start_argument_index, end_argument_index, predicate_index, suitable_verb):
		subwords, token_start_idxs = self.subword_tokenize(tokens, start_argument_index, end_argument_index, predicate_index, suitable_verb)
		start_argument_index = torch.tensor([subwords.index(self.START_ARG)])
		end_argument_index = torch.tensor([subwords.index(self.END_ARG)])
		predicate_index = torch.tensor([token_start_idxs[predicate_index]])
		subword_ids, mask = self.convert_tokens_to_ids(subwords)

		return subword_ids, mask, start_argument_index, end_argument_index, predicate_index