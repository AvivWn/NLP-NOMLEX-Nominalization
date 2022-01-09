from typing import Set, List, Dict
from dataclasses import dataclass, field
from itertools import chain

from yet_another_verb.arguments_extractor.extraction.extracted_argument import ExtractedArgument, ArgRange
from yet_another_verb.nomlex.constants import ArgumentType
from yet_another_verb.nomlex.representation.constraints_map import ConstraintsMap


@dataclass
class Extraction:
	words: list
	predicate_idx: int
	predicate_lemma: str
	args: Set[ExtractedArgument]
	fulfilled_constraints: List[ConstraintsMap] = field(default_factory=list, compare=False)
	arg_by_range: Dict[ArgRange, ExtractedArgument] = field(default_factory=dict, compare=False)
	arg_by_type: Dict[ArgumentType, ExtractedArgument] = field(default_factory=dict, compare=False)

	def __post_init__(self):
		for arg in self.args:
			self.arg_by_type[arg.arg_type] = arg
			self.arg_by_range[arg.tightest_range] = arg

	def __len__(self):
		return len(self.args)

	@property
	def arg_indices(self) -> Set[int]:
		return set(chain(*[arg.arg_idxs for arg in self.args]))

	@property
	def arg_types(self) -> Set[str]:
		return set([arg.arg_type for arg in self.args])

	def tag_arg_by_range(self, idx_range: ArgRange, tag: str):
		arg = self.arg_by_range.get(idx_range, None)

		if arg is not None:
			arg.arg_tag = tag

	def tag_arg_by_type(self, arg_type: ArgumentType, tag: str):
		arg = self.arg_by_type.get(arg_type, None)

		if arg is not None:
			arg.arg_tag = tag


Extractions = List[Extraction]
