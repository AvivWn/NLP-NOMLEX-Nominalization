import torch
from transformers import BertModel, BertTokenizerFast, RobertaModel, RobertaTokenizerFast
from argparse import ArgumentParser, Namespace

from noun_as_verb.constants.dataset_constants import NOUNS_TAG, ARGS_TAG

class PretrainedWrapper:
	MASK = '[MASK]'
	CLS = "[CLS]"
	SEP = "[SEP]"
	START_ARG = "<arg>"
	END_ARG = "</arg>"
	START_PRED = "<pred>"
	END_PRED = "</pred>"

	TAGSET_ENCODER = {ARGS_TAG: 0, NOUNS_TAG: 1}

	def __init__(self, hyper_params:Namespace):
		super().__init__()
		print(hyper_params)
		self.model_name = hyper_params.pretrained_encoder
		self.lower_case = "uncased" in self.model_name

		if self.model_name.startswith("bert"):
			# BERT
			self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name, do_lower_case=self.lower_case)
			self.pretrained_model = BertModel.from_pretrained(self.model_name)
		else:
			# RoBERTa
			self.tokenizer = RobertaTokenizerFast.from_pretrained(self.model_name)
			self.pretrained_model = RobertaModel.from_pretrained(self.model_name)
			self.CLS = "<s>"
			self.SEP = "</s>"

		# Add the new tokens to BERT
		self.new_tags = [self.START_ARG, self.END_ARG, self.START_PRED, self.END_PRED]
		n_new_tags = self.tokenizer.add_tokens(self.new_tags)
		assert len(self.new_tags) == n_new_tags, "Couldn't add all the new tokens!"
		self.pretrained_model.resize_token_embeddings(len(self.tokenizer))

		self.max_len = 100 #self.pretrained_model.embeddings.position_embeddings.weight.size(0)
		self.dim = self.pretrained_model.embeddings.position_embeddings.weight.size(1)

	@staticmethod
	def add_encoder_specific_args(parent_parser):
		parser = ArgumentParser(parents=[parent_parser], add_help=False)
		parser.add_argument('--pretrained_encoder', type=str, default='bert-base-uncased') # roberta-base
		return parser

	def get_max_len(self):
		return self.max_len

	def get_dim(self):
		return self.dim

	def get_model(self):
		return self.pretrained_model

	def convert_tokens_to_ids(self, tokens):
		token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
		ids = torch.tensor(token_ids)

		padded_ids = torch.zeros(self.max_len).to(ids)
		padded_ids[:len(ids)] = ids
		mask = torch.zeros(self.max_len).to(ids)
		mask[:len(ids)] = 1
		return padded_ids, mask

	def insert_tags(self, words, arg_start_idx, arg_end_idx, predicate_idx):
		assert predicate_idx >= arg_end_idx or predicate_idx < arg_start_idx, "The predicate was found inside an argument!"
		assert set(self.new_tags).isdisjoint(words), "The sentence already includes the new special tags!"

		# Add just the predicate tags
		# if arg_start_idx == -1:
		# 	words.insert(predicate_idx + 1, self.END_PRED)
		# 	words.insert(predicate_idx, self.START_PRED)
		# 	return

		if arg_start_idx == -1:
			words.insert(predicate_idx + 1, self.END_PRED)
			words.insert(predicate_idx + 1, self.END_ARG)
			words.insert(predicate_idx, self.START_ARG)
			words.insert(predicate_idx, self.START_PRED)
			return

		# argument before predicate
		if arg_end_idx <= predicate_idx:
			words.insert(predicate_idx + 1, self.END_PRED)
			words.insert(predicate_idx, self.START_PRED)
			words.insert(arg_end_idx, self.END_ARG)
			words.insert(arg_start_idx, self.START_ARG)
		else: # predicate before argument
			words.insert(arg_end_idx, self.END_ARG)
			words.insert(arg_start_idx, self.START_ARG)
			words.insert(predicate_idx + 1, self.END_PRED)
			words.insert(predicate_idx, self.START_PRED)

		return

	def encode(self, words, arg_start_idx, arg_end_idx, predicate_idx, verb, tagset_type, context, all_sizes=False):
		"""
		Inserts new special tokens in the given indexes, to the sentence string
		And encodes all the tokens in the sentence using Pre-Trained tokenizer
		:param words: list of words in a sentence
		:param arg_start_idx: index of the start of an argument, might be -1 if the argument is None
		:param arg_end_idx: index of the end of an argument, might be -1 if the argument is None
		:param predicate_idx: index of a specific predicate in sentence
		:param verb: the verb that associated with the given predicate
		:param tagset_type: the type of the tagset (ARGS or NOUNS). Every tagset has a bit different encoding method
		:return: the encoded sentence, mask and indexes to the special tokens within
		"""

		assert (arg_end_idx == -1) ^ (arg_start_idx != -1), "Illegal argument indexes!"
		#assert (tagset_type == ARGS_TAG) ^ (arg_end_idx == -1 and arg_start_idx == -1), f"Example input isn't compatible with tagset type! {tagset_type, arg_start_idx, arg_end_idx}"

		tagset_id = torch.tensor(self.TAGSET_ENCODER[tagset_type])

		# without any context, ignore the sentence except the argument
		if not context:
			words = words[arg_start_idx:arg_end_idx]

		else:
			self.insert_tags(words, arg_start_idx, arg_end_idx, predicate_idx)
			verb = verb if tagset_type == ARGS_TAG else None

		tokens = self.tokenizer.tokenize(" ".join(words), pair=verb, add_special_tokens=True)

		# Ignore examples with too many tokens
		if len(tokens) >= self.max_len and not all_sizes:
			return None

		# if tagset_type == ARGS_TAG:
		# 	start_arg_idx = torch.tensor(tokens.index(self.START_ARG))
		# 	end_arg_idx = torch.tensor(tokens.index(self.END_ARG))
		# 	verb_idx = torch.tensor(tokens.index(self.SEP) + 1)
		# else:
		# 	start_arg_idx = torch.tensor(-1)
		# 	end_arg_idx = torch.tensor(-1)
		# 	verb_idx = torch.tensor(-1)

		start_arg_idx = torch.tensor(tokens.index(self.START_ARG) if self.START_ARG in tokens else -1)
		end_arg_idx = torch.tensor(tokens.index(self.END_ARG) if self.END_ARG in tokens else -1)

		start_predicate_idx = torch.tensor(tokens.index(self.START_PRED) if self.START_PRED in tokens else -1)
		end_predicate_idx = torch.tensor(tokens.index(self.END_PRED) if self.END_PRED in tokens else -1)

		verb_idx = torch.tensor(tokens.index(self.SEP) + 1)

		subword_ids, mask = self.convert_tokens_to_ids(tokens)

		return subword_ids, mask, start_arg_idx, end_arg_idx, \
			   start_predicate_idx, end_predicate_idx, verb_idx, tagset_id

	def encode_without_context(self, words, verb):
		tokens = self.tokenizer.tokenize(" ".join(words), pair=verb, add_special_tokens=True)

		# Ignore examples with too many tokens
		if len(tokens) >= self.max_len:
			return None

		subword_ids, mask = self.convert_tokens_to_ids(tokens)
		return subword_ids, mask