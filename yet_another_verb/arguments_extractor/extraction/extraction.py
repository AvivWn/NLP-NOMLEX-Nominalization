from typing import *
from dataclasses import dataclass, field
from itertools import chain

from yet_another_verb.arguments_extractor.extraction.extracted_argument import ExtractedArgument
from yet_another_verb.nomlex.representation.constraints_map import ConstraintsMap


@dataclass
class Extraction:
	predicate_idx: int
	args: Set[ExtractedArgument]
	fulfilled_constraints: List[ConstraintsMap] = field(default_factory=list, compare=False)

	def __len__(self):
		return len(self.args)

	@property
	def arg_indices(self) -> Set[int]:
		return set(chain(*[arg.arg_idxs for arg in self.args]))


Extractions = List[Extraction]
