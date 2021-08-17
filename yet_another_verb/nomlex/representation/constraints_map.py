from typing import List, Optional

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from yet_another_verb.nomlex.constants import ArgumentType, WordRelation, POSTag


# class OnRequiredValue()


@dataclass_json
@dataclass
class ConstraintsMap:
	arg_type: Optional[ArgumentType] = field(default=None)
	controlled: List[ArgumentType] = field(default_factory=list)

	word_relations: List[WordRelation] = field(default_factory=list)
	postags: List[POSTag] = field(default_factory=list)
	values: List[str] = field(default_factory=list)
	attributes: List[str] = field(default_factory=list)
	required: bool = field(default=True)
	plural: bool = field(default=False)
	subjunct: bool = field(default=False)

	relatives_constraints: List['ConstraintsMap'] = field(default_factory=list)

	def __hash__(self):
		return hash(str(self))
