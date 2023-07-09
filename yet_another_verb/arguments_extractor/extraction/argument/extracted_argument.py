from typing import Optional, List, Tuple, Union, TYPE_CHECKING, Iterable
from dataclasses import dataclass, field

from yet_another_verb.arguments_extractor.extraction.utils.indices import get_indices_as_range
from yet_another_verb.exceptions import EmptyArgumentException
from yet_another_verb.nomlex.representation.constraints_map import ConstraintsMap
from yet_another_verb.arguments_extractor.extraction.argument.argument_type import ArgumentType

if TYPE_CHECKING:
	from yet_another_verb.sentence_encoding.encoding import Encoding


ArgRange = Tuple[int, int]


@dataclass
class ExtractedArgument:
	start_idx: int
	end_idx: int
	head_idx: Optional[int] = field(default=None)
	arg_type: Optional[ArgumentType] = field(default=None)
	arg_tag: Optional[Union[str, ArgumentType]] = field(default=None)
	fulfilled_constraints: List[ConstraintsMap] = field(default_factory=list, compare=False)
	encoding: Optional['Encoding'] = field(default=None)

	def __post_init__(self):
		self.arg_tag = self.arg_type

	def __hash__(self):
		return hash((self.arg_type, self.start_idx, self.end_idx, self.head_idx))

	def __setattr__(self, key: str, value):
		super().__setattr__(key, value)

		if key == 'arg_type':
			self.arg_tag = value

	@property
	def arg_indices(self) -> List[int]:
		return list(range(self.start_idx, self.end_idx + 1))

	def add_indices(self, arg_indices: Iterable[int]):
		arg_indices = list(arg_indices) + [self.start_idx, self.end_idx]
		self.start_idx, self.end_idx = get_indices_as_range(arg_indices)

	def remove_indices(self, arg_indices: Iterable[int]):
		new_arg_indices = set(self.arg_indices) - set(arg_indices)

		if len(new_arg_indices) == 0:
			raise EmptyArgumentException("Trying to remove all indices from an argument, presumably because arguments overlap.")

		self.start_idx, self.end_idx = get_indices_as_range(new_arg_indices)


ExtractedArguments = List[ExtractedArgument]
