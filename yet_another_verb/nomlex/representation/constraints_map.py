from typing import List, Optional
from itertools import chain

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from yet_another_verb.dependency_parsing import DepRelation, POSTag
from yet_another_verb.arguments_extractor.extraction.argument.argument_type import ArgumentType


@dataclass_json
@dataclass
class ConstraintsMap:
	arg_type: Optional[ArgumentType] = field(default=None)

	dep_relations: List[DepRelation] = field(default_factory=list)
	postags: List[POSTag] = field(default_factory=list)
	values: List[str] = field(default_factory=list)
	required: bool = field(default=True)

	relatives_constraints: List['ConstraintsMap'] = field(default_factory=list)

	def _update_included_args(self):
		included_args = set(chain(*[c.included_args for c in self.relatives_constraints]))

		if self.arg_type is not None:
			included_args.add(self.arg_type)

		self._included_args = sorted(list(included_args))

	def __setattr__(self, key: str, value: List['ConstraintsMap']):
		super().__setattr__(key, value)

		if key == 'relatives_constraints':
			self._update_included_args()

	def __hash__(self):
		return hash(str(self))

	@property
	def included_args(self) -> List[ArgumentType]:
		return self._included_args

	def get_required_relative_maps(self) -> List['ConstraintsMap']:
		return [relative_map for relative_map in self.relatives_constraints if relative_map.required]


ORConstraintsMaps = List[ConstraintsMap]  # (constraint 1) OR (constraint 2) OR (...)
ANDConstraintsMaps = List[ConstraintsMap]  # (constraint 1) AND (constraint 2) AND (...)
