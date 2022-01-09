from typing import Dict
from dataclasses import dataclass
from itertools import chain

from yet_another_verb.arguments_extractor.extraction.extraction import Extractions


@dataclass
class MultiWordExtraction:
	words: list
	extractions_per_idx: Dict[int, Extractions]

	@property
	def extractions(self):
		return list(chain(*[e for e in self.extractions_per_idx.values()]))

	def update(self, more_extractions_per_idx: Dict[int, Extractions]):
		self.extractions_per_idx.update(more_extractions_per_idx)
