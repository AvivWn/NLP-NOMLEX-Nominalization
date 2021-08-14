from typing import *
from dataclasses import dataclass

from yet_another_verb.arguments_extractor.extraction.extracted_argument import ExtractedArgument


@dataclass
class Extraction:
	predicate_idx: int
	args: List[ExtractedArgument]

	def __len__(self):
		return len(self.args)
