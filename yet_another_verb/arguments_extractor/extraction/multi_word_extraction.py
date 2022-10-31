from typing import Dict, List, TYPE_CHECKING
from dataclasses import dataclass
from itertools import chain

from yet_another_verb.arguments_extractor.extraction import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.extraction import Extractions

if TYPE_CHECKING:
	from yet_another_verb.arguments_extractor.extraction.words import Words


@dataclass
class MultiWordExtraction:
	words: 'Words'
	extractions_per_idx: Dict[int, Extractions]

	@property
	def extractions(self) -> Extractions:
		return list(chain(*[e for e in self.extractions_per_idx.values()]))

	@property
	def all_args(self) -> List[ExtractedArgument]:
		return list(chain(*[e.all_args for e in self.extractions]))

	def update(self, more_extractions_per_idx: Dict[int, Extractions]):
		self.extractions_per_idx.update(more_extractions_per_idx)


MultiWordExtractions = List[MultiWordExtraction]
