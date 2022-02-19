from typing import Iterator

from tqdm import tqdm

from yet_another_verb import ArgsExtractor
from yet_another_verb.arguments_extractor.extraction.multi_word_extraction import MultiWordExtractions
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.file_handlers.pkl_file_handler import PKLFileHandler
from yet_another_verb.file_handlers.parsed_bin_file_handler import ParsedBinFileHandler
from yet_another_verb.ml.datasets_creator.dataset_creator import DatasetCreator


class ExtractedDatasetCreator(DatasetCreator):
	def __init__(
			self, in_dataset_path: str,
			args_extractor: ArgsExtractor, dependency_parser: DependencyParser,
			dataset_size=None, **kwargs
	):
		super().__init__(dataset_size)
		self.in_dataset_path = in_dataset_path
		self.dependency_parser = dependency_parser
		self.args_extractor = args_extractor

	def _sents_to_extractions(self, docs: Iterator[ParsedText]) -> MultiWordExtractions:
		multi_word_extractions = []
		total_extracted_predicates = []

		for doc in tqdm(docs):
			multi_word_extraction = self.args_extractor.extract_multiword(doc)
			extracted_predicates = multi_word_extraction.extractions_per_idx.keys()

			if len(extracted_predicates) == 0:
				continue

			multi_word_extractions.append(multi_word_extraction)
			total_extracted_predicates += extracted_predicates

			if self.has_reached_size(total_extracted_predicates):
				break

		return multi_word_extractions

	def create_dataset(self, out_dataset_path):
		parsed_bin = ParsedBinFileHandler(self.dependency_parser).load(self.in_dataset_path)
		in_dataset = parsed_bin.get_parsed_texts()
		out_dataset = self._sents_to_extractions(in_dataset)
		PKLFileHandler.save(out_dataset_path, out_dataset)
