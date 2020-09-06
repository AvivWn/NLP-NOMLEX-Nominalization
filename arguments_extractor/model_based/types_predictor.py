import os
from os.path import join
from collections import defaultdict

import torch
import numpy as np
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from spacy.tokens import Token

from arguments_extractor.model_based.types_model import TypesModel, CheckpointEveryNSteps
from arguments_extractor.model_based.pretrained_wrapper import PretrainedWrapper
from arguments_extractor.constants.lexicon_constants import *
from arguments_extractor.utils import difference_list, reverse_dict
from arguments_extractor import config

class TypesPredictor:
	LR = 2 * pow(10, -5)
	WEIGHT_DECAY = 2e-3
	BATCH_SIZE = 128
	TRAIN_TEST_RATIO = 0.8	# with regard to the number of verbs
	PRETRAINED_ENCODER = "bert-base-uncased"  # roberta-base
	TRAINED_MODEL_PATH = os.path.dirname(__file__) + "/trained_model.pth"
	ARGS_TAGSET = {COMP_OBJ: 0, COMP_SUBJ: 1, COMP_IND_OBJ: 2, COMP_PP: 3, COMP_NONE: 4}
	# ARGS_TAGSET = {COMP_OBJ: 0, COMP_SUBJ: 1, COMP_PP: 2, COMP_NONE: 3}
	# NOUNS_TAGSET = {NOM_TYPE_OBJ: 0, NOM_TYPE_SUBJ: 1, NOM_TYPE_IND_OBJ: 2, NOM_TYPE_VERB_NOM: 3, "NONE": 4}
	NOUNS_TAGSET = {NOM_TYPE_OBJ: 0, NOM_TYPE_SUBJ: 1, NOM_TYPE_VERB_NOM: 2, "NONE": 3}
	# NOUNS_TAGSET = {NOM_TYPE_SUBJ: 0, NOM_TYPE_VERB_NOM: 1, "NONE": 2}
	REVERSE_ARGS_TAGSET = reverse_dict(ARGS_TAGSET)
	REVERSE_NOUNS_TAGSET = reverse_dict(NOUNS_TAGSET)
	TAGSETS = {"ARGS": (ARGS_TAGSET, REVERSE_ARGS_TAGSET),
			   "NOUNS": (NOUNS_TAGSET, REVERSE_NOUNS_TAGSET)}

	pretrained_wrapper: PretrainedWrapper # Wraps a pretrained model
	model: TypesModel # The model

	def __init__(self):
		use_cuda = torch.cuda.is_available()
		gpus = -1 if use_cuda else None

		self.pretrained_wrapper = PretrainedWrapper(self.PRETRAINED_ENCODER)
		self.model = TypesModel(self.ARGS_TAGSET, self.NOUNS_TAGSET, self.pretrained_wrapper, lr=2 * (10 ** -5), weight_decay=2e-3)

		tb_logger = TensorBoardLogger(join(config.DATASETS_PATH, "lightning_logs"), name='')
		self.pl_trainer = Trainer(max_epochs=3, gpus=gpus, distributed_backend="dp", val_check_interval=0.1,
						  deterministic=True, logger=tb_logger, callbacks=[CheckpointEveryNSteps(1000)])

	def train(self):
		self.pl_trainer.fit(self.model)

	def test(self):
		self.pl_trainer.test()

	def load_model(self):
		# ckpt_path = self.pl_trainer.checkpoint_callback.best_model_path
		# self.model = TypesModel.load_from_checkpoint(ckpt_path)
		self.model.eval()

	def get_tagset(self, tagset_type):
		return self.TAGSETS[tagset_type]

	def predict(self, dependency_tree, candidate_start, candidate_end, predicate_idx, suitable_verb, tagset_type, limited_types=None):
		tokens = [token.orth_ for token in dependency_tree]
		dataset, reverse_dataset = self.get_tagset(tagset_type)

		# if candidate_index == predicate_index:
		# 	argument_start_index = candidate_index
		# 	argument_end_index = candidate_index
		# else:
		# 	argument_span = list(dependency_tree[candidate_index].subtree)
		# 	argument_start_index = argument_span[0].i
		# 	argument_end_index = argument_span[-1].i

		*features, tagset_id = self.pretrained_wrapper.encode(tokens, candidate_start, candidate_end,
																predicate_idx, suitable_verb, tagset_type)

		with torch.no_grad():
			output = self.model(features, tagset_id).view(-1)

		# Avoid impossible predictions
		if limited_types is not None:
			excluded_ids = [dataset[arg_type] for arg_type in difference_list(dataset, limited_types)]
			output[excluded_ids] = -np.inf

		logits = F.log_softmax(output, dim=0)
		predicted_type = reverse_dataset[logits.argmax().item()]

		logits[-np.inf == logits] = 0
		entropy = -(np.exp(logits) * logits).sum()

		return predicted_type, entropy

	# def determine_args_type(self, args_per_candidate, predicate_token, suitable_verb, default_subcat=False):
	# 	uncertain_candidates = {COMP_SUBJ: [], COMP_OBJ: [], COMP_IND_OBJ: [], COMP_PP: [], COMP_PP1: [], COMP_PP2: []}
	# 	updated_args_per_candidate = defaultdict(list)
	#
	# 	# Choose for each candidate token, the most appropriate complement type (using the model)
	# 	for candidate_token, matched_arguments in args_per_candidate.items():
	# 		complement_types = set([argument.get_real_complement_type() for argument in matched_arguments])
	#
	# 		if complement_types.issubset(set(uncertain_candidates.keys())):
	# 			if len(complement_types) > 1:
	# 				# complements types with PP instead of PP1 or PP2
	# 				tmp_complement_types = complement_types
	# 				if COMP_PP1 in complement_types or COMP_PP2 in complement_types:
	# 					tmp_complement_types = list(set(difference_list(complement_types, [COMP_PP1, COMP_PP2]) + [COMP_PP]))
	#
	# 				# Predict the most compatible complement type, using the model
	# 				predicted_complement_type, entropy = self.predict(candidate_token.doc, candidate_token.i, predicate_token.i, suitable_verb, "ARG", limited_types=tmp_complement_types)
	#
	# 				if predicted_complement_type == COMP_NONE:
	# 					continue
	#
	# 				# Get the suitable predicted complement type (PP might mean PP1 or PP2)
	# 				suitable_complement_type = predicted_complement_type
	# 				if predicted_complement_type == COMP_PP:
	# 					if COMP_PP1 in complement_types: suitable_complement_type = COMP_PP1
	# 					elif COMP_PP2 in complement_types: suitable_complement_type = COMP_PP2
	#
	# 				# Find the matched arguments of that complement type and save it in the uncertain cadidates
	# 				matched_arguments = [argument for argument in matched_arguments if argument.get_real_complement_type() == suitable_complement_type]
	# 				uncertain_candidates[suitable_complement_type].append((candidate_token, entropy, matched_arguments))
	#
	# 			elif len(complement_types) == 1:
	# 				uncertain_candidates[list(complement_types)[0]].append((candidate_token, np.nan, matched_arguments))
	#
	# 		else:
	# 			updated_args_per_candidate[candidate_token] = difference_list(matched_arguments, uncertain_candidates.keys())
	#
	# 	# Now, determine for each complement type, the most appropriate candidates (using entropy)
	# 	for complement_type, candidates in uncertain_candidates.items():
	# 		if candidates == []:
	# 			continue
	#
	# 		# If there is only one option and it isn't the default subcat
	# 		if len(candidates) == 1 and not default_subcat:
	# 			candidate_token, entropy, matched_arguments = candidates[0]
	# 			updated_args_per_candidate[candidate_token] = matched_arguments
	# 			continue
	#
	# 		# Calculate the entropy for the candidates that didn't passed through the model
	# 		# These are the candidates that had only one possible complement type
	# 		new_candidates = []
	# 		for i, candidate in enumerate(candidates):
	# 			candidate_token, entropy, matched_arguments = candidate
	#
	# 			if entropy == np.nan:
	# 				predicted_complement_type, entropy = self.predict(candidate_token.doc, candidate_token.i, predicate_token.i, suitable_verb, "ARGS")
	#
	# 				# Save this candidate only if the model was sure that this candidate gets this appropriate complement type
	# 				if predicted_complement_type != COMP_NONE and predicted_complement_type == complement_type.replace(COMP_PP1, COMP_PP).replace(COMP_PP2, COMP_PP):
	# 					new_candidates.append((candidate_token, entropy, matched_arguments))
	# 			else:
	# 				new_candidates.append(candidate)
	#
	# 		if len(new_candidates) < 1:
	# 			continue
	#
	# 		# Sort candidates by entropy (The lower the entropy, the higher the certainty)
	# 		new_candidates.sort(key=lambda candidate: candidate[1])
	#
	# 		# Choose the best candidate
	# 		candidate_token, _, matched_arguments = new_candidates[0]
	# 		updated_args_per_candidate[candidate_token] = matched_arguments
	#
	# 		# The maximum number of compatible PP is 2 for the default subcat
	# 		if complement_type == COMP_PP and len(new_candidates) > 1 and default_subcat:
	# 			candidate_token, _, matched_arguments = new_candidates[1]
	# 			updated_args_per_candidate[candidate_token] += matched_arguments
	#
	# 	return updated_args_per_candidate

	def choose_arg_type(self, candidate: Token, args: list, types: set, predicate: Token, verb: str, default_subcat=False):
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
		candidate_start, candidate_end = args[0].get_range_idxs()
		predicted_type, entropy = self.predict(candidate.doc, candidate_start, candidate_end, predicate.i, verb, "ARGS", limited_types=types)

		# Find the arguments of the predicted types
		get_type = lambda arg: arg.get_real_type().replace("1", "").replace("2", "") # PP1 and PP2 are like PP
		args = [a for a in args if get_type(a) == predicted_type]
		return predicted_type, entropy, args

	def determine_args_type(self, candidates_args, predicate: Token, verb, default_subcat=False):
		uncertain_types = [COMP_OBJ, COMP_SUBJ, COMP_IND_OBJ, COMP_PP, COMP_PP1, COMP_PP2]
		uncertain_candidates = defaultdict(list)
		determined_dict = defaultdict(list) # The determined arguments for each candidate

		# Each candidate should take one appropriate type, determined by the model
		for candidate, args in candidates_args.items():
			types = set([a.get_real_type() for a in args])

			# The candidate is compatible with some "certain" complements
			if not types.issubset(uncertain_types):
				determined_dict[candidate] = [a for a in args if a.get_real_type() not in uncertain_types]
				continue

			# Find the appropriate type and add this candidate to its list of options
			predicted_type, entropy, args = self.choose_arg_type(candidate, args, types, predicate, verb, default_subcat)
			if args != []:
				uncertain_candidates[predicted_type].append((candidate, entropy, args))

		# Now, determine for each complement type, the most appropriate candidates (using entropy)
		for arg_type, candidates_info in uncertain_candidates.items():
			candidates_info.sort(key=lambda c_info: c_info[1])

			# Choose the best candidate
			candidate, _, args = candidates_info[0]
			determined_dict[candidate] = args

			# The maximum number of compatible PP is 2 for the default subcat
			if arg_type == COMP_PP and len(candidates_info) > 1 and default_subcat:
				candidate, _, args = candidates_info[1]
				determined_dict[candidate] = args

		return determined_dict

	def determine_noun_type(self, noun: Token):
		predicted_type, _ = self.predict(noun.doc, -1, -1, noun.i, "", "NOUNS", limited_types=None)
		return predicted_type