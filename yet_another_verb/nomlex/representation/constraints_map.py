from typing import List, Set, Optional
from itertools import chain

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from yet_another_verb.nomlex.constants import ArgumentType, WordRelation, POSTag


@dataclass_json
@dataclass
class ConstraintsMap:
	arg_type: Optional[ArgumentType] = field(default=None)

	word_relations: List[WordRelation] = field(default_factory=list)
	postags: List[POSTag] = field(default_factory=list)
	values: List[str] = field(default_factory=list)
	required: bool = field(default=True)

	included_args: Set[ArgumentType] = field(default_factory=set)
	relatives_constraints: List['ConstraintsMap'] = field(default_factory=list)

	def _update_included_args(self):
		self.included_args.update(chain(*[c.included_args for c in self.relatives_constraints]))

		if self.arg_type is not None:
			self.included_args.add(self.arg_type)

	def __post_init__(self):
		self._update_included_args()

	def __setattr__(self, key: str, value: List['ConstraintsMap']):
		super().__setattr__(key, value)

		if key == 'relatives_constraints':
			self.__post_init__()

	def __hash__(self):
		return hash(str(self))

	def get_required_relative_maps(self) -> List['ConstraintsMap']:
		return [relative_map for relative_map in self.relatives_constraints if relative_map.required]


ORConstraintsMaps = List[ConstraintsMap]  # (constraint 1) OR (constraint 2) OR (...)
ANDConstraintsMaps = List[ConstraintsMap]  # (constraint 1) AND (constraint 2) AND (...)
