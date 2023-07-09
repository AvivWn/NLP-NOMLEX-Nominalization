from collections import defaultdict
from itertools import combinations, permutations
from typing import List, Dict, Set, Tuple

from dataclasses import dataclass, field

import numpy as np

from yet_another_verb.arguments_extractor.extraction import Extraction
from yet_another_verb.arguments_extractor.extraction.argument.argument_type import ArgumentTypes, ArgumentType


@dataclass
class ReferencesCorpus:
	extractions: List[Extraction]
	encodings_by_arg_type: Dict[str, np.array]
	ext_indices_by_arg_type: Dict[str, np.array]
	arg_types_combos: Dict[int, Set[Tuple[str]]] = field(default_factory=dict)

	def __post_init__(self):
		self.arg_types_combos = defaultdict(set)

		for ext in self.extractions:
			for n_args in range(1, len(ext.arg_types) + 1):
				self.arg_types_combos[n_args].update(combinations(ext.arg_types, n_args))

	def get_ordered_arg_types_combos(self, n_args: int):
		closest_n_args = min(max(self.arg_types_combos.keys()), n_args)

		ordered_combinations = set()
		for combo in self.arg_types_combos[closest_n_args]:
			filled_combo = list(combo) + [ArgumentType.REDUNDANT] * (n_args - closest_n_args)
			ordered_combinations.update(permutations(filled_combo))

		return list(ordered_combinations)

	def get_filtered_references(self, references_amount: int, arg_types: ArgumentTypes):
		extractions = self.extractions
		encodings_by_arg_type = self.encodings_by_arg_type
		ext_indices_by_arg_type = self.ext_indices_by_arg_type

		if references_amount is not None:
			extractions = extractions[:references_amount]

			for arg_type in ext_indices_by_arg_type:
				ext_indices = ext_indices_by_arg_type[arg_type]
				ext_indices_by_arg_type[arg_type] = ext_indices[ext_indices < references_amount]
				encodings_by_arg_type[arg_type] = encodings_by_arg_type[arg_type][:len(ext_indices_by_arg_type[arg_type])]
				assert len(ext_indices_by_arg_type[arg_type]) == len(encodings_by_arg_type[arg_type])

		if arg_types is not None:
			encodings_by_arg_type = {arg_type: encs for arg_type, encs in encodings_by_arg_type.items() if arg_type in arg_types}

		references_corpus = ReferencesCorpus(extractions, encodings_by_arg_type, ext_indices_by_arg_type)
		return references_corpus


ReferencesByPredicate = Dict[str, ReferencesCorpus]
