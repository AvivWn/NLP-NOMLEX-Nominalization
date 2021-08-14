from typing import List, Dict

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


from yet_another_verb.nomlex.constants import SubcatType
from yet_another_verb.nomlex.representation.constraints_map import ConstraintsMap


@dataclass_json
@dataclass
class LexicalEntry:
	orth: str
	related_orths: List[str] = field(default_factory=list)
	subcats: Dict[SubcatType, List[ConstraintsMap]] = field(default_factory=dict)
