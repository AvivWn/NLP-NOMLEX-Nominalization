from typing import List, Dict, Iterator

from itertools import chain
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from yet_another_verb.dependency_parsing.dependency_parser.parsed_word import ParsedWord
from yet_another_verb.nomlex.constants import SubcatType
from yet_another_verb.nomlex.representation.constraints_map import ConstraintsMap
from yet_another_verb.nomlex.representation.indexed_contraints import IndexedConstraintsMaps
from yet_another_verb.nomlex.representation.lexical_subcat import LexicalSubcat


@dataclass_json
@dataclass
class LexicalEntry:
	orth: str
	related_orths: List[str]  # orths in the same word-family
	ambiguous_forms: List[str]  # forms that might be misleading
	subcats: Dict[SubcatType, LexicalSubcat] = field(default_factory=dict)

	def get_constraints_maps(self) -> List[ConstraintsMap]:
		return list(set(chain(*[s.constraints_maps for s in self.subcats.values()])))

	def __hash__(self):
		return hash(self.orth)


@dataclass_json
@dataclass
class CombinedLexicalEntry:
	orth: str
	entries: List[LexicalEntry]

	_indexed_constrints_maps: IndexedConstraintsMaps = field(default=None, compare=False, repr=False)

	def index_constraints_maps(self):
		constraints_maps = list(set(chain(*[e.get_constraints_maps() for e in self.entries])))
		self._indexed_constrints_maps = IndexedConstraintsMaps(constraints_maps)

	def get_relevant_constraints_maps(self, word: ParsedWord, order_by_potential: bool) -> Iterator['ConstraintsMap']:
		return self._indexed_constrints_maps.get_relevant_constraints_maps(word, order_by_potential)

	def get_related_orths(self) -> List[str]:
		return list(chain(*[e.related_orths for e in self.entries]))
