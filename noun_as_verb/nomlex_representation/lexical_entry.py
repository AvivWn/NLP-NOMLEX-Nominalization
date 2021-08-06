from typing import List, Dict

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


from noun_as_verb.nomlex_representation.lexicon_constants import SubcatType
from noun_as_verb.nomlex_representation.constraints_map import ConstraintsMap


@dataclass_json
@dataclass
class LexicalEntry:
	orth: str
	related_orths: List[str] = field(default_factory=list)
	subcats: Dict[SubcatType, List[ConstraintsMap]] = field(default_factory=dict)
