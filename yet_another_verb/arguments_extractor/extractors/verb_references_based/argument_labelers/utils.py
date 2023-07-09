from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional

import numpy as np

from yet_another_verb.arguments_extractor.extraction import ExtractedArguments, ArgumentType
from yet_another_verb.arguments_extractor.extractors.verb_references_based.similarity_scorer.utils import \
	SimilaritiesByArg

TypesDistribution = Dict[ArgumentType, float]
TypesDistributions = List[TypesDistribution]


def get_k_largests_indices(values, kth):
	kth = min(len(values), kth)
	return np.argpartition(-np.array(values), range(kth))[:kth]


def label_redundant_by_threshold(args: ExtractedArguments, similarities_by_arg: SimilaritiesByArg, threshold=None):
	if threshold is None:
		return []

	redundant_args = []
	for arg in args:
		similarities = list(chain(*similarities_by_arg[arg].values()))
		most_matching_score = np.max(similarities)

		if most_matching_score < threshold:
			redundant_args.append(arg)
			arg.arg_type = ArgumentType.REDUNDANT

	return redundant_args


def label_args_by_distributions(
		args: ExtractedArguments, type_distributions: TypesDistributions, max_repeated_args: Optional[int] = None):
	type_distribution_by_arg = {}
	for arg, type_dist in zip(args, type_distributions):
		type_distribution_by_arg[arg] = type_dist

	args_by_type = defaultdict(list)

	while len(args) > 0:
		arg = args[0]
		type_distribution = type_distribution_by_arg[arg]
		matching_type = ArgumentType.REDUNDANT

		while matching_type == ArgumentType.REDUNDANT:
			if len(type_distribution) == 0:
				break

			matching_type = max(type_distribution, key=type_distribution.get)
			matching_similarity = type_distribution[matching_type]

			if matching_type is ArgumentType.REDUNDANT:
				break

			if max_repeated_args is not None and len(args_by_type[matching_type]) == max_repeated_args:
				matched_similarities = [type_distribution_by_arg[matched_arg][matching_type] for matched_arg in
										args_by_type[matching_type]]
				most_unsimilar_idx = np.argmin(matched_similarities)

				if matching_similarity > matched_similarities[most_unsimilar_idx]:
					args.append(args_by_type[matching_type].pop(most_unsimilar_idx))
				else:
					type_distribution.pop(matching_type)
					matching_type = ArgumentType.REDUNDANT

		args_by_type[matching_type].append(arg)
		args = args[1:]

	for arg_type, args in args_by_type.items():
		for arg in args:
			arg.arg_type = arg_type
