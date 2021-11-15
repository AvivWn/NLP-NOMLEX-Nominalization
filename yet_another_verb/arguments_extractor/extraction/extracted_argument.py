from typing import Set, Optional, List
from dataclasses import dataclass, field

from yet_another_verb.nomlex.constants import ArgumentType
from yet_another_verb.nomlex.representation.constraints_map import ConstraintsMap


@dataclass
class ExtractedArgument:
	arg_idxs: Set[int]
	arg_type: Optional[ArgumentType] = field(default=None)
	fulfilled_constraints: List[ConstraintsMap] = field(default_factory=list, compare=False)

	def __hash__(self):
		return hash((self.arg_type, tuple(self.arg_idxs)))
