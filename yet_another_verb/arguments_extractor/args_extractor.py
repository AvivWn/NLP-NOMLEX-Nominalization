import abc
from typing import List, Optional, Dict

from yet_another_verb.arguments_extractor.extraction.extraction import Extraction


class ArgsExtractor(abc.ABC):
	@abc.abstractmethod
	def preprocess(self, text) -> list:
		pass

	@abc.abstractmethod
	def extract(self, word_idx: int, preprocessed_text: list) -> List[Extraction]:
		pass

	def extract_multiword(self, preprocessed_text, limited_idxs: Optional[list] = None) -> Dict[int, List[Extraction]]:
		extractions_per_idx = {}

		for i in range(len(preprocessed_text)):
			if limited_idxs is None or i in limited_idxs:
				extractions_per_idx[i] = self.extract(i, preprocessed_text)

		return extractions_per_idx
