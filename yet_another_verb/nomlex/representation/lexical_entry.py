from typing import List, Dict

from itertools import chain
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from functools import lru_cache

from yet_another_verb.nomlex.constants import SubcatType
from yet_another_verb.nomlex.representation.constraints_map import ConstraintsMap
from yet_another_verb.nomlex.representation.lexical_subcat import LexicalSubcat


@dataclass_json
@dataclass
class LexicalEntry:
	orth: str
	related_orths: List[str] = field(default_factory=list)
	subcats: Dict[SubcatType, LexicalSubcat] = field(default_factory=dict)

	def __hash__(self):
		return hash(self.orth)

	@property
	@lru_cache(maxsize=None)
	def constraints_maps(self) -> List[ConstraintsMap]:
		maps = list(set(chain(*[s.constraints_maps for s in self.subcats.values()])))
		maps.sort(key=lambda m: len(m.included_args), reverse=True)
		return maps
