import torch
import numpy as np
from transformers import BertModel, BertTokenizerFast, RobertaModel, RobertaTokenizerFast

from arguments_extractor.utils import flatten

class PretrainedWrapper:
	MASK = '[MASK]'
	CLS = "[CLS]"
	SEP = "[SEP]"
	START_ARG = "<ARG>"
	END_ARG = "</ARG>"

	TAGSET_ENCODER = {"ARGS": 0, "NOUNS": 1}

	def __init__(self, model_name):
		super().__init__()
		self.model_name = model_name

		if model_name.startswith("bert"):
			# BERT
			do_lower_case = "uncased" in model_name
			self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name, do_lower_case=do_lower_case)
			self.pretrained_model = BertModel.from_pretrained(self.model_name)
		else:
			# RoBERTa
			self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
			self.pretrained_model = RobertaModel.from_pretrained(model_name)
			self.CLS = "<s>"
			self.SEP = "</s>"

		# Freeze all the weights, except to the last layer
		for param in self.pretrained_model.parameters():
			param.requires_grad = False

		for param in list(self.pretrained_model.children())[-1].parameters():
			param.requires_grad = True

		self.max_len = self.pretrained_model.embeddings.position_embeddings.weight.size(0)
		self.dim = self.pretrained_model.embeddings.position_embeddings.weight.size(1)

	def convert_tokens_to_ids(self, tokens):
		token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
		ids = torch.tensor([token_ids])
		assert ids.size(1) < self.max_len

		padded_ids = torch.zeros(1, self.max_len).to(ids)
		padded_ids[0, :ids.size(1)] = ids
		mask = torch.zeros(1, self.max_len).to(ids)
		mask[0, :ids.size(1)] = 1
		return padded_ids, mask

	def subword_tokenize(self, tokens, arg_start_idx, arg_end_idx, predicate_idx, verb):
		subwords = list(map(self.tokenizer.tokenize, tokens))
		subword_lengths = [0] + list(map(len, subwords))

		if arg_start_idx != -1 and arg_end_idx != -1:
			subwords = [self.CLS] + flatten(subwords[:arg_start_idx]) + \
					   [self.START_ARG] + flatten(subwords[arg_start_idx:arg_end_idx+1]) + [self.END_ARG] + \
					   flatten(subwords[arg_end_idx+1:]) + \
					   [self.SEP] + list(self.tokenizer.tokenize(verb)) + [self.SEP]
						# flatten(subwords[end_argument_index+1:]) + [self.SEP] + subwords[predicate_idx] + [self.SEP]

			subword_lengths[arg_end_idx + 1] += 1
			subword_lengths[arg_start_idx] += 1
		else:
			subwords = [self.CLS] + flatten(subwords[:]) + [self.SEP]

		token_start_idxs = 1 + np.cumsum(subword_lengths[:-1])
		return subwords, token_start_idxs

	def encode(self, tokens, arg_start_idx, arg_end_idx, predicate_idx, verb, tagset_type):
		"""

		:param tokens:
		:param arg_start_idx:
		:param arg_end_idx:
		:param predicate_idx:
		:param verb:
		:param tagset_type:
		:return:
		"""

		subwords, token_start_idxs = self.subword_tokenize(tokens, arg_start_idx, arg_end_idx, predicate_idx, verb)

		if arg_start_idx != -1 and arg_end_idx != -1:
			start_arg_idx = torch.tensor([subwords.index(self.START_ARG)])
			end_arg_idx = torch.tensor([subwords.index(self.END_ARG)])
		else:
			start_arg_idx = torch.tensor([-1])
			end_arg_idx = torch.tensor([-1])

		tagset_id = torch.tensor([self.TAGSET_ENCODER[tagset_type]])
		predicate_idx = torch.tensor([token_start_idxs[predicate_idx]])
		verb_idx = torch.tensor([subwords.index(self.SEP) + 1])
		subword_ids, mask = self.convert_tokens_to_ids(subwords)

		return subword_ids, mask, start_arg_idx, end_arg_idx, \
			   predicate_idx, verb_idx, tagset_id