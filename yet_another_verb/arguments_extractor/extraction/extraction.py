from typing import Set, List, Dict, Optional
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
	typeless_args: Set[ExtractedArgument] = field(default_factory=set, compare=False)
	undetermined_args: Set[ExtractedArgument] = field(default_factory=set, compare=False)

	arg_by_range: Dict[ArgRange, ExtractedArgument] = field(default_factory=dict, compare=False)
	arg_by_type: Dict[ArgumentType, ExtractedArgument] = field(default_factory=dict, compare=False)

	def __post_init__(self):
		typed_args = set()
		for arg in self.args:
			if arg.arg_type is None:
				self.typeless_args.add(arg)
			else:
				typed_args.add(arg)
				self.arg_by_type[arg.arg_type] = arg
				self.arg_by_range[arg.tightest_range] = arg

		self.args = typed_args

	def __len__(self):
		return len(self.args)

	@property
	def arg_indices(self) -> Set[int]:
		return set(chain(*[arg.arg_idxs for arg in self.args]))

	@property
	def arg_types(self) -> Set[str]:
		return set([arg.arg_type for arg in self.args])

	@property
	def fulfilled_constraints(self) -> List[ConstraintsMap]:
		total_args = set.union(self.args, self.typeless_args)
		return list(chain(*[arg.fulfilled_constraints for arg in total_args]))

	@property
	def predicate_type(self) -> Optional[ArgumentType]:
		if self.predicate_idx not in self.arg_indices:
			return None

		predicate_args = [arg for arg in self.args if self.predicate_idx in arg.arg_idxs]
		assert len(predicate_args) <= 1

		if len(predicate_args) == 0:
			return None

		return predicate_args[0].arg_type

	def tag_arg_by_range(self, idx_range: ArgRange, tag: str):
		arg = self.arg_by_range.get(idx_range, None)

		if arg is not None:
			arg.arg_tag = tag

	def tag_arg_by_type(self, arg_type: ArgumentType, tag: str):
		arg = self.arg_by_type.get(arg_type, None)

		if arg is not None:
			arg.arg_tag = tag


Extractions = List[Extraction]
