from typing import Dict, List

from yet_another_verb.arguments_extractor.args_extractor import ArgsExtractor
from yet_another_verb.arguments_extractor.extraction.extraction import Extraction


class MultiwordArgumentsExtractor:
	def __init__(self, args_extractor: ArgsExtractor):
		self.args_extractor = args_extractor

	def preprocess_text(self, text) -> list:
		return self.args_extractor.preprocess(text)

	def extract(self, preprocessed_text: list) -> Dict[int, List[Extraction]]:
		extractions_per_idx = {}

		for i in range(len(preprocessed_text)):
			extractions_per_idx[i] = self.args_extractor.extract(i, preprocessed_text)

		return extractions_per_idx
