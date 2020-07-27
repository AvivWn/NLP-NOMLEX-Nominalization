import os
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch import optim

from arguments_extractor.model_based.arguments_model import ArgumentsModel
from arguments_extractor.learning_process.training import training
from arguments_extractor.model_based.pretrained_encoder import PretrainedEncoder
from arguments_extractor.constants.lexicon_constants import COMP_SUBJ, COMP_OBJ, COMP_IND_OBJ, COMP_PP, COMP_PP1, COMP_PP2
from arguments_extractor.utils import difference_list, reverse_dict

class ArgumentsPredictor:
	LR = 3 * pow(10, -5)
	WEIGHT_DECAY = 1e-3
	BATCH_SIZE = 128
	PRETRAINED_ENCODER = "bert-base-uncased"  # roberta-base
	TAGSET = {COMP_OBJ: 0, COMP_SUBJ: 1, COMP_IND_OBJ: 2, COMP_PP: 3}

	pretrained_encoder: PretrainedEncoder
	model: ArgumentsModel

	def __init__(self):
		self.trained_model_path = os.path.dirname(__file__) + "/trained_model.pth"
		self.REVERSE_TAGSET = reverse_dict(self.TAGSET)

		self.pretrained_encoder = PretrainedEncoder(self.PRETRAINED_ENCODER)
		self.model = ArgumentsModel(len(self.TAGSET), self.pretrained_encoder)

	def _load_dataset(self, dataset_path):
		dataset = pd.read_csv(dataset_path, sep="\t", index_col=0)

		# Example
		# dataset = pd.DataFrame([("The appointement of the man", 2, 4, 1, "appoint", "OBJECT"),
		# 						("The man appointed him", 0, 1, 2, "appoint", "SUBJECT"),
		# 						("The man appointed him", 3, 3, 2, "appoint", "OBJECT")])

		dataset_tuples = []

		for row_info in tqdm(dataset.itertuples(), "Loading the dataset", leave=False):
			_, sentence, argument_start_index, argument_end_index, predicate_index, suitable_verb, label = tuple(row_info)
			suitable_verb = suitable_verb.split("#")[0]
			tokens = sentence.split(" ")
			label_id = torch.tensor([self.TAGSET[label]])

			bert_ids, bert_mask, start_argument_index, end_argument_index, predicate_index = self.pretrained_encoder.subword_tokenize_to_ids(tokens, argument_start_index, argument_end_index, predicate_index, suitable_verb)
			dataset_tuples.append((bert_ids, bert_mask, start_argument_index, end_argument_index, predicate_index, label_id))

		# Build the data-loader object
		dataset = TensorDataset(*tuple(map(torch.cat, zip(*dataset_tuples))))
		dataloader = DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=True)

		return dataloader

	def train(self, train_set_path, test_set_path):
		train_dataloader = self._load_dataset(train_set_path)
		test_dataloader = self._load_dataset(test_set_path)

		optimizer = optim.Adam(self.model.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY)
		training(train_dataloader, test_dataloader, self.REVERSE_TAGSET, self.model, optimizer, self.trained_model_path)

	def load_model(self):
		use_cuda = torch.cuda.is_available()
		device = torch.device("cuda:0" if use_cuda else "cpu")
		self.model.load_state_dict(torch.load(self.trained_model_path, map_location=device))
		self.model.eval()

	def predict(self, dependency_tree, candidate_index, predicate_index, suitable_verb, limited_complements=None):
		tokens = [token.orth_ for token in dependency_tree]

		if candidate_index == predicate_index:
			argument_start_index = candidate_index
			argument_end_index = candidate_index
		else:
			argument_span = list(dependency_tree[candidate_index].subtree)
			argument_start_index = argument_span[0].i
			argument_end_index = argument_span[-1].i

		enconder_output = self.pretrained_encoder.subword_tokenize_to_ids(tokens, argument_start_index, argument_end_index, predicate_index, suitable_verb)

		excluded_tagset_ids = None
		if limited_complements is not None:
			excluded_tagset_ids = [self.TAGSET[complement_type] for complement_type in difference_list(self.TAGSET, limited_complements)]

		with torch.no_grad():
			logits = self.model(*enconder_output, excluded_tagset_ids=excluded_tagset_ids).detach().cpu().numpy()[0]

		predicted_complement_type = self.REVERSE_TAGSET[logits.argmax()]

		logits[-np.inf == logits] = 0
		entropy = -(np.exp(logits) * logits).sum()

		return predicted_complement_type, entropy

	def determine_args_type(self, args_per_candidate, predicate_token, suitable_verb, default_subcat=False):
		uncertain_candidates = {COMP_SUBJ: [], COMP_OBJ: [], COMP_IND_OBJ: [], COMP_PP: [], COMP_PP1: [], COMP_PP2: []}
		updated_args_per_candidate = defaultdict(list)

		# Choose for each candidate token, the most appropriate complement type (using the model)
		for candidate_token, matched_arguments in args_per_candidate.items():
			complement_types = set([argument.get_real_complement_type() for argument in matched_arguments])

			if complement_types.issubset(set(uncertain_candidates.keys())):
				if len(complement_types) > 1:
					# complements types with PP instead of PP1 or PP2
					tmp_complement_types = complement_types
					if COMP_PP1 in complement_types or COMP_PP2 in complement_types:
						tmp_complement_types = list(set(difference_list(complement_types, [COMP_PP1, COMP_PP2]) + [COMP_PP]))

					# Predict the most compatible complement type, using the model
					predicted_complement_type, entropy = self.predict(candidate_token.doc, candidate_token.i, predicate_token.i, suitable_verb, limited_complements=tmp_complement_types)

					# Get the suitable predicted complement type (PP might mean PP1 or PP2)
					suitable_complement_type = predicted_complement_type
					if predicted_complement_type == COMP_PP:
						if COMP_PP1 in complement_types: suitable_complement_type = COMP_PP1
						elif COMP_PP2 in complement_types: suitable_complement_type = COMP_PP2

					# Find the matched arguments of that complement type and save it in the uncertain cadidates
					matched_arguments = [argument for argument in matched_arguments if argument.get_real_complement_type() == suitable_complement_type]
					uncertain_candidates[suitable_complement_type].append((candidate_token, entropy, matched_arguments))

				elif len(complement_types) == 1:
					uncertain_candidates[list(complement_types)[0]].append((candidate_token, np.nan, matched_arguments))

			else:
				updated_args_per_candidate[candidate_token] = difference_list(matched_arguments, uncertain_candidates.keys())

		# Now, determine for each complement type, the most appropriate candidates (using entropy)
		for complement_type, candidates in uncertain_candidates.items():
			if candidates == []:
				continue

			# If there is only one option
			if len(candidates) == 1:
				candidate_token, entropy, matched_arguments = candidates[0]
				updated_args_per_candidate[candidate_token] = matched_arguments
				continue

			# Calculate the entropy for the candidates that didn't passed through the model
			# These are the candidates that had only one possible complement type
			new_candidates = []
			for i, candidate in enumerate(candidates):
				candidate_token, entropy, matched_arguments = candidate

				if entropy == np.nan:
					predicted_complement_type, entropy = self.predict(candidate_token.doc, candidate_token.i, predicate_token.i, suitable_verb)

					# Save this candidate only if the model was sure that this candidate gets this appropriate complement type
					if predicted_complement_type == complement_type.replace(COMP_PP1, COMP_PP).replace(COMP_PP2, COMP_PP):
						new_candidates.append((candidate_token, entropy, matched_arguments))
				else:
					new_candidates.append(candidate)

			if len(new_candidates) < 1:
				continue

			# Sort candidates by entropy (The lower the entropy, the higher the certainty)
			new_candidates.sort(key=lambda candidate: candidate[1])

			# Choose the best candidate
			candidate_token, _, matched_arguments = new_candidates[0]
			updated_args_per_candidate[candidate_token] = matched_arguments

			# The maximum number of compatible PP is 2 for the default subcat
			if complement_type == COMP_PP and len(new_candidates) > 1 and default_subcat:
				candidate_token, _, matched_arguments = new_candidates[1]
				updated_args_per_candidate[candidate_token] += matched_arguments

		return updated_args_per_candidate