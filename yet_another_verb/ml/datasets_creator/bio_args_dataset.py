from typing import List
from collections import namedtuple

from yet_another_verb.arguments_extractor.extraction.multi_word_extraction import MultiWordExtractions
from yet_another_verb.arguments_extractor.extraction.representation.str_bio_representation import StrBIORepresentation
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.file_handlers import PKLFileHandler, CSVFileHandler
from yet_another_verb.ml.datasets_creator.dataset_creator import DatasetCreator

BIOTaggedSentence = namedtuple("BIOTaggedSentence", "word_idx words tags")


class BIOArgsDatasetCreator(DatasetCreator):
	def __init__(self, in_dataset_path: str, dataset_size=None, limited_postags=None, limited_types=None, **kwargs):
		super().__init__(dataset_size)
		self.in_dataset_path = in_dataset_path
		self.limited_postags = limited_postags
		self.limited_types = limited_types
		self.bio_representation = StrBIORepresentation()

	def extractions_to_bio_tags(self, multi_word_extractions: MultiWordExtractions) -> List[BIOTaggedSentence]:
		bios = []

		for multi_word_extraction in multi_word_extractions:
			words = multi_word_extraction.words

			if not isinstance(words, ParsedText):
				raise Exception("Cannot generate dataset from unparsed sentence.")

			bio_by_idx_repr = self.bio_representation.represent_combined_dict(
				multi_word_extraction, arg_types=self.limited_types, safe_combine=True
			)

			for predicate_idx, bio_repr in bio_by_idx_repr.items():
				if self.limited_postags is None or words[predicate_idx].tag in self.limited_postags:
					bios.append(BIOTaggedSentence(predicate_idx, words, bio_repr))

		return bios

	def create_dataset(self, out_dataset_path: str):
		in_dataset = PKLFileHandler.load(self.in_dataset_path)
		out_dataset = self.extractions_to_bio_tags(in_dataset)
		CSVFileHandler.save(out_dataset_path, out_dataset)
