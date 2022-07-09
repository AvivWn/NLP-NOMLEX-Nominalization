from typing import Dict, List

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from yet_another_verb.nomlex.representation.lexical_entry import CombinedLexicalEntry


@dataclass_json
@dataclass
class Lexicon:
	entries: Dict[str, CombinedLexicalEntry] = field(default_factory=dict)  # multiple entries for the same orth form

	def get_related_orths(self, orth: str) -> List[str]:
		return self.entries[orth].get_related_orths()

	def enhance_orths(self, orths: List[str]):
		more_orths = []
		for orth in orths:
			more_orths += self.get_related_orths(orth)

		orths += more_orths
