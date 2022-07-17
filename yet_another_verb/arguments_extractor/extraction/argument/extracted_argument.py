from typing import Set, Optional, List, Tuple, Union
from dataclasses import dataclass, field

from yet_another_verb.nomlex.representation.constraints_map import ConstraintsMap
from yet_another_verb.arguments_extractor.extraction.argument.argument_type import ArgumentType

ArgRange = Tuple[int, int]


@dataclass
class ExtractedArgument:
	arg_idxs: Set[int]
	arg_type: Optional[ArgumentType] = field(default=None)
	arg_tag: Optional[Union[str, ArgumentType]] = field(default=None)
	fulfilled_constraints: List[ConstraintsMap] = field(default_factory=list, compare=False)

	def __post_init__(self):
		self.arg_tag = self.arg_type

	def __hash__(self):
		return hash((self.arg_type, tuple(self.arg_idxs)))

	@property
	def tightest_range(self) -> ArgRange:
		return min(self.arg_idxs), max(self.arg_idxs)
