import abc
from typing import List

from yet_another_verb.arguments_extractor.extraction.extraction import Extraction


class ArgsExtractor(abc.ABC):
	@abc.abstractmethod
	def preprocess(self, text) -> list:
		pass

	@abc.abstractmethod
	def extract(self, word_idx: int, preprocessed_text: list) -> List[Extraction]:
		pass
