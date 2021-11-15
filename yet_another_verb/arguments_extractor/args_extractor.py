import abc
from typing import Optional, Dict

from yet_another_verb.arguments_extractor.extraction.extraction import Extractions


class ArgsExtractor(abc.ABC):
	@abc.abstractmethod
	def preprocess(self, text) -> list:
		pass

	@abc.abstractmethod
	def extract(self, word_idx: int, preprocessed_text: list) -> Optional[Extractions]:
		pass

	def extract_multiword(self, preprocessed_text, limited_idxs: Optional[list] = None) -> Dict[int, Extractions]:
		extractions_per_idx = {}

		for i in range(len(preprocessed_text)):
			if limited_idxs is None or i in limited_idxs:
				extractions = self.extract(i, preprocessed_text)

				if extractions is not None:
					extractions_per_idx[i] = extractions

		return extractions_per_idx
