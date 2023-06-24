from typing import Set, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from itertools import chain

from yet_another_verb.arguments_extractor.extraction.argument.extracted_argument import ExtractedArgument, ArgRange
from yet_another_verb.arguments_extractor.extraction.argument.argument_type import ArgumentType
from yet_another_verb.nomlex.representation.constraints_map import ConstraintsMap

if TYPE_CHECKING:
	from yet_another_verb.arguments_extractor.extraction.words import Words


@dataclass
class Extraction:
	words: 'Words'
	predicate_idx: int
	predicate_lemma: str
	predicate_postag: str  # VERB, NOUN, ...
	args: List[ExtractedArgument]
	undetermined_args: List[ExtractedArgument] = field(default_factory=list, compare=False)
	fulfilled_constraints: ConstraintsMap = field(default=None, compare=False)

	@staticmethod
	def _sorted_args_by_indices(args: List[ExtractedArgument]) -> List[ExtractedArgument]:
		return sorted(args, key=lambda arg: (arg.start_idx, arg.end_idx))

	def _seperate_typeless_args(self, args: List[ExtractedArgument]) -> tuple:
		typed_args, typeless_args = [], []
		for arg in args:
			if arg.arg_type is not None:
				typed_args.append(arg)
			else:
				typeless_args.append(arg)

		typed_args = self._sorted_args_by_indices(typed_args)
		typeless_args = self._sorted_args_by_indices(typeless_args)
		return typed_args, typeless_args

	def _update_arg_mapping(self):
		self._arg_by_type, self._arg_by_range = {}, {}
		for arg in self.args:
			self._arg_by_type[arg.arg_type] = arg
			self._arg_by_range[(arg.start_idx, arg.end_idx)] = arg

	def __setattr__(self, key: str, value: List[ExtractedArgument]):
		if key == 'args':
			value, typeless_args = self._seperate_typeless_args(value)
			self._typeless_args = typeless_args
			super().__setattr__(key, value)
			self._update_arg_mapping()
			return

		if key == 'undetermined_args':
			value = self._sorted_args_by_indices(value)

		super().__setattr__(key, value)

	def __len__(self):
		return len(self.args)

	@property
	def arg_indices(self) -> Set[int]:
		return set(chain(*[arg.arg_indices for arg in self.args]))

	@property
	def arg_types(self) -> Set[str]:
		return set([arg.arg_type for arg in self.args])

	@property
	def typeless_args(self) -> List[ExtractedArgument]:
		return self._typeless_args

	@property
	def all_args(self) -> List[ExtractedArgument]:
		return self.args + self.undetermined_args + self.typeless_args

	@property
	def fulfilled_constraints_by_args(self) -> List[ConstraintsMap]:
		total_args = self.args + list(self._typeless_args)
		return list(chain(*[arg.fulfilled_constraints for arg in total_args]))

	@property
	def predicate_arg(self) -> Optional[ExtractedArgument]:
		predicate_args = [arg for arg in self.args if self.predicate_idx in arg.arg_indices]
		assert len(predicate_args) <= 1

		if len(predicate_args) == 0:
			return None

		return predicate_args[0]

	@property
	def predicate_type(self) -> Optional[ArgumentType]:
		predicate_arg = self.predicate_arg

		if predicate_arg is None:
			return None

		return predicate_arg.arg_type

	def tag_arg_by_range(self, idx_range: ArgRange, tag: str):
		arg = self._arg_by_range.get(idx_range, None)

		if arg is not None:
			arg.arg_tag = tag

	def tag_arg_by_type(self, arg_type: ArgumentType, tag: str):
		arg = self._arg_by_type.get(arg_type, None)

		if arg is not None:
			arg.arg_tag = tag

	def get_arg_by_range(self, idx_range: ArgRange) -> Optional[ExtractedArgument]:
		return self._arg_by_range.get(idx_range)

	def get_arg_by_type(self, arg_type: ArgumentType) -> Optional[ExtractedArgument]:
		return self._arg_by_type.get(arg_type)


Extractions = List[Extraction]
