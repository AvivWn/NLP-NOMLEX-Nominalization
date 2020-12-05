import os
from os.path import join
from collections import defaultdict
from argparse import ArgumentParser
from itertools import permutations

import torch
import numpy as np
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from spacy.tokens import Token, Span

from noun_as_verb.model_based.types_model import TypesModel, CheckpointEveryNSteps
from noun_as_verb.model_based.pretrained_wrapper import PretrainedWrapper
from noun_as_verb.rule_based import ExtractedArgument
from noun_as_verb.constants.lexicon_constants import *
from noun_as_verb.utils import difference_list, reverse_dict, is_proper_noun
from noun_as_verb import config

class TypesPredictor:
	"""
		Determines the appropriate type (role-type) for verb's and nom's arguments
	"""

	def __init__(self, role_types:set):
		parser = ArgumentParser()
		parser = TypesModel.add_model_specific_args(parser)
		parser = PretrainedWrapper.add_encoder_specific_args(parser)
		self.hyper_params, _ = parser.parse_known_args()

		# if self.hyper_params.tagset.lower() in ["semantic", "sem"]:
		# 	args_labels = SEM_ARGS_LABELS
		# 	nouns_labels = SEM_NOUNS_LABELS
		# else: # syntactic or syn
		# 	args_labels = SYN_ARGS_LABELS
		# 	nouns_labels = SYN_NOUNS_LABELS
		#
		# self.args_tagset, self.reverse_args_tagset = self.labels_to_tagset(args_labels)
		# self.nouns_tagset, self.reverse_nouns_tagset = self.labels_to_tagset(nouns_labels)

		self.tagset, self.reverse_tagset = self.labels_to_tagset(role_types)
		self.hyper_params.tagset = self.tagset
		self.model = TypesModel(self.hyper_params)

	def train(self):
		parser = ArgumentParser()
		parser = Trainer.add_argparse_args(parser)
		trainer_args, _ = parser.parse_known_args()

		data_version = self.hyper_params.data_version
		experiment_version = self.hyper_params.experiment_version
		log_dir = join(config.DATASETS_PATH, "lightning_logs", data_version, experiment_version)
		counted_log_dir = log_dir

		i = 0
		while os.path.isdir(counted_log_dir):
			i += 1
			counted_log_dir = log_dir + str(i)

		experiment_version += str(i) if i > 0 else ""
		tb_logger = TensorBoardLogger(join(config.DATASETS_PATH, "lightning_logs"), name=data_version, version=experiment_version)
		stopping = EarlyStopping(monitor='loss/val', min_delta=0.00, patience=5, verbose=False, mode='min')
		pl_trainer = Trainer.from_argparse_args(trainer_args, logger=tb_logger, callbacks=[CheckpointEveryNSteps(1000)], early_stop_callback=stopping)

		pl_trainer.fit(self.model)

	def load_model(self):
		# Loads a pretrained model from the best checkpoint (currently the model is saved locally)
		# @TODO- update the path to an downlable link, when the package is ready

		# @TODO- update ckpt path, and update gpu usage if possible
		ckpt_path = join(config.DATASETS_PATH, "verb_args/base/model.ckpt")
		if os.path.isfile(ckpt_path):
			self.model = self.model.load_from_checkpoint(ckpt_path)
			#self.model = TypesModel.load_from_checkpoint(ckpt_path)

		#device = "cuda" if torch.cuda.is_available() else "cpu"
		#self.model = self.model.to(torch.device(device))
		#print(self.model.device)

		self.model.eval()



	@staticmethod
	def labels_to_tagset(labels):
		# returns a deterministic tagset dictionary, given a list of labels
		assert len(set(labels)) == len(labels)
		reverse_tagset = dict(enumerate(sorted(labels)))
		tagset = reverse_dict(reverse_tagset)
		return tagset, reverse_tagset

	# def get_tagset(self, tagset_type):
	# 	if tagset_type == NOUNS_TAG:
	# 		return self.nouns_tagset, self.reverse_nouns_tagset
	# 	elif tagset_type == ARGS_TAG:
	# 		return self.args_tagset, self.reverse_args_tagset

	def predict(self, dependency_tree, candidate_start, candidate_end, predicate_idx, suitable_verb, tagset_type, limited_types=None):
		tokens = [token.orth_ for token in dependency_tree]

		*features, _ = self.model.encode(tokens, candidate_start, candidate_end,
												 predicate_idx, suitable_verb, tagset_type, all_sizes=True)

		with torch.no_grad():
			output = self.model(features).view(-1)

		# Avoid impossible predictions
		if limited_types is not None:
			excluded_ids = [self.tagset[arg_type] for arg_type in difference_list(self.tagset, limited_types)]
			output[excluded_ids] = -np.inf

		logits = F.log_softmax(output, dim=0)
		#predicted_type = self.reverse_dataset[logits.argmax().item()]
		#print(tokens[predicate_idx], tokens[candidate_start:candidate_end], predicted_type)

		#logits[-np.inf == logits] = 0
		#entropy = -(np.exp(logits) * logits).sum()

		#return predicted_type, entropy
		return logits

	def get_types_distribution(self, candidate: Span, types: set, predicate: Token, verb: str, default_subcat=False):
		# Chooses the most appropriate argument type for the given candidate

		# Single type candidate
		if len(types) == 1:
			if not default_subcat:
				return types.pop(), 0 # The highest certainty

			# For the default subcat, even single choice is uncertain
			types.add(COMP_NONE)

		# complements types with PP instead of PP1 or PP2
		if not {COMP_PP1, COMP_PP2}.isdisjoint(types):
			types.difference_update([COMP_PP1, COMP_PP2])
			types.add(COMP_PP)

		# Predict the most compatible complement type, using the model
		candidate_start, candidate_end = candidate[0].i, candidate[-1].i + 1
		logits = self.predict(candidate.doc, candidate_start, candidate_end, predicate.i, verb, "ARGS", limited_types=types)

		# Find the arguments of the predicted types
		#get_type = lambda arg: arg.get_real_type().replace("1", "").replace("2", "") # PP1 and PP2 are like PP
		return logits

	def choose_arg_type2(self, candidate: Span, args: list, types: set, predicate: Token, verb: str, default_subcat=False):
		# Chooses the most appropriate argument type for the given candidate

		# Single type candidate
		if len(types) == 1:
			if not default_subcat:
				return types.pop(), 0, args # The highest certainty

			# For the default subcat, even single choice is uncertain
			types.add(COMP_NONE)

		# complements types with PP instead of PP1 or PP2
		if {COMP_PP1, COMP_PP2}.isdisjoint(types):
			types.difference_update([COMP_PP1, COMP_PP2])
			types.add(COMP_PP)

		# Predict the most compatible complement type, using the model
		candidate_start, candidate_end = candidate[0].i, candidate[-1].i + 1
		predicted_type, entropy = self.predict(candidate.doc, candidate_start, candidate_end, predicate.i, verb, "ARGS", limited_types=types)

		# Find the arguments of the predicted types
		get_type = lambda arg: arg.get_real_type().replace("1", "").replace("2", "") # PP1 and PP2 are like PP
		args = [a for a in args if get_type(a) == predicted_type]
		return predicted_type, entropy, args

	def determine_args_type(self, candidates_args, predicate: ExtractedArgument, verb, default_subcat=False):
		# Determines the most appropriate type of each candidate, using a model

		uncertain_types = list(self.tagset.keys()) + ([COMP_PP1, COMP_PP2] if COMP_PP in self.tagset else [])
		uncertain_candidates = {}
		predicate_token = predicate.get_token()
		determined_dict = {}
		none_spans = []

		# Each candidate should take one appropriate type, determined by the model
		for candidate_span, role_types in candidates_args.items():
			role_types = set(role_types)

			if predicate.get_token().i == candidate_span[0].i or role_types.isdisjoint(uncertain_types):
				determined_dict[candidate_span] = role_types
				continue

			if candidate_span.lemma_ in ["i", "he", "she", "it", "they", "we", "-PRON-"]:
				determined_dict[candidate_span] = role_types
				continue

			role_types.add(COMP_NONE)
			logits = self.get_types_distribution(candidate_span, role_types, predicate_token, verb, default_subcat)

			if logits.argmax().item() == self.tagset[COMP_NONE]:
				none_spans.append(candidate_span)
			else:
				uncertain_candidates[candidate_span] = logits

		if len(uncertain_candidates) == 0:
			return determined_dict

		#print(dict(candidates_args))

		# if uncertain_candidates == {}:
		# 	return {}

		u = list(uncertain_candidates.keys())
		u += [None] * (len(uncertain_types) - 2) #(len(self.tagset) - 1 - len(uncertain_candidates))

		certain_types = [] #[list(types)[0] for types in determined_dict.values() if len(types) == 1]
		role_types = difference_list(uncertain_types, [COMP_NONE] + certain_types)

		#if len(predicate_types) == 1:
		#	role_types = difference_list(role_types, predicate_types)

		types_combinations = list(permutations(u, len(role_types)))
		empty_comb = tuple([None] * len(role_types))
		if empty_comb not in types_combinations:
			types_combinations.append(empty_comb)

		#print(predicate.get_token(), types_combinations)

		args_sum_logits = []

		for comb in types_combinations:
			# sum_logits = 0
			# for i, arg in enumerate(comb):
			# 	if arg:
			# 		print(i, role_types[i], uncertain_candidates[arg][self.tagset[role_types[i]]])
			# 		sum_logits += uncertain_candidates[arg][self.tagset[role_types[i]]].item()

			sum_logits = sum([uncertain_candidates[arg][self.tagset[role_types[i]]] for i, arg in enumerate(comb) if arg])
			sum_logits += sum([uncertain_candidates[arg][self.tagset[COMP_NONE]].item() for arg in set(u).difference(comb) if arg])
			args_sum_logits.append(sum_logits)

		#print(predicate.get_token(), args_sum_logits)
		max_idx = int(np.argmax(args_sum_logits))
		best = types_combinations[max_idx]

		determined_dict.update({arg: [role_types[i]] for i, arg in enumerate(best) if arg})

		for arg in difference_list(candidates_args.keys(), determined_dict.keys()):
			determined_dict[arg] = difference_list(candidates_args[arg], uncertain_types)

		#if predicate_span:
		#	determined_dict[predicate_span] = predicate_types

		#assert all([set(determined_dict[s]).isdisjoint(uncertain_types) for s in none_spans])

		#print(predicate.get_token(), len(types_combinations), determined_dict)
		return determined_dict

	def determine_args_type2(self, candidates_args, predicate: ExtractedArgument, verb, default_subcat=False):
		# Determines the most appropriate type of each candidate, using a model

		uncertain_types = list(self.tagset.keys()) + ([COMP_PP1, COMP_PP2] if COMP_PP in self.tagset else [])
		uncertain_candidates = defaultdict(list)
		determined_dict = defaultdict(list) # The determined arguments for each candidate
		predicate_token = predicate.get_token()

		# Each candidate should take one appropriate type, determined by the model
		for candidate, args in candidates_args.items():
			types = set([a.get_real_type() for a in args])
			candidate_span = args[0].as_span(trim_argument=False)

			# The candidate is compatible with some "certain" complements
			if not types.issubset(uncertain_types):
				determined_dict[candidate] = [a for a in args if a.get_real_type() not in uncertain_types]
				continue

			# Find the appropriate type and add this candidate to its list of options
			predicted_type, entropy, args = self.choose_arg_type(candidate_span, args, types, predicate_token, verb, default_subcat)
			if args != []:
				uncertain_candidates[predicted_type].append((candidate, entropy, args))

		# The predicate might also take an argument role
		predicate_type = predicate.get_name()
		if predicate_type:
			uncertain_candidates[predicate_type].append((predicate_token, predicate.get_entropy(), [predicate]))

		# Now, determine for each complement type, the most appropriate candidates (using entropy)
		for arg_type, candidates_info in uncertain_candidates.items():
			candidates_info.sort(key=lambda c_info: c_info[1])

			# Choose the best candidate
			candidate, _, args = candidates_info[0]
			determined_dict[candidate] = args
			print(candidate, [arg.get_token() for arg in args])

			# The maximum number of compatible PP is 2 for the default subcat
			if arg_type == COMP_PP and len(candidates_info) > 1 and default_subcat:
				candidate, _, args = candidates_info[1]
				determined_dict[candidate] = args

		return determined_dict

	def determine_noun_type(self, noun: Token):
		# Determines the most appropriate type of the given noun, using a model

		# Proper nouns are considered as common nouns, which cannot be noms
		# The model wasn't trained over proper nouns
		if is_proper_noun(noun):
			return None, None

		# Otherwise, use a model to determine
		predicted_type, entropy = self.predict(noun.doc, -1, -1, noun.i, "", "NOUNS", limited_types=None)
		return predicted_type, entropy