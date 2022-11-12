from itertools import chain
from typing import Iterator, Set, List

from tqdm import tqdm

from yet_another_verb.arguments_extractor.args_extractor import ArgsExtractor
from yet_another_verb.arguments_extractor.extraction import MultiWordExtractions
from yet_another_verb.data_handling.bytes.compressed.compressed_parsed_text import CompressedParsedText
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.data_handling import ParsedBinFileHandler, ExtractedFileHandler
from yet_another_verb.data_handling.dataset_creator import DatasetCreator


class ExtractedFromParsedDatasetCreator(DatasetCreator):
	def __init__(
			self, in_dataset_path: str,
			args_extractor: ArgsExtractor,
			dataset_size=None, **kwargs
	):
		super().__init__(dataset_size)
		self.in_dataset_path = in_dataset_path
		self.args_extractor = args_extractor

	@staticmethod
	def _get_all_extracted_sentences(multi_word_extractions: MultiWordExtractions):
		extracted_sentences = set()
		for ext in multi_word_extractions:
			if isinstance(ext.words, ParsedText):
				extracted_sentences.add(ext.words.tokenized_text)
			elif isinstance(ext.words, CompressedParsedText):
				extracted_sentences.add(ext.words.parsed_text.tokenized_text)
			elif len(ext.words) > 0 and isinstance(ext.words[0], str):
				extracted_sentences.add(" ".join(ext.words))

		return extracted_sentences

	def _sents_to_extractions(
			self, docs: Iterator[ParsedText],
			ignored_sentences: Set[str] = None, extracted_predicates: List[int] = None) -> MultiWordExtractions:
		total_extracted_predicates = [] if extracted_predicates is None else extracted_predicates
		multi_word_extractions = []

		for doc in tqdm(docs, leave=False):
			if self.has_reached_size(total_extracted_predicates):
				break

			if ignored_sentences is not None and doc.tokenized_text in ignored_sentences:
				continue

			multi_word_extraction = self.args_extractor.extract_multiword(doc)
			extracted_predicates = multi_word_extraction.extractions_per_idx.keys()

			if len(extracted_predicates) == 0:
				continue

			multi_word_extractions.append(multi_word_extraction)
			total_extracted_predicates += extracted_predicates

		return multi_word_extractions

	def append_dataset(self, out_dataset_path: str):
		multi_word_extractions: MultiWordExtractions = ExtractedFileHandler(keep_compressed=True).load(out_dataset_path)
		extracted_sentences = self._get_all_extracted_sentences(multi_word_extractions)
		extracted_predicates = list(chain(*[ext.extractions_per_idx.keys() for ext in multi_word_extractions]))

		in_parsed_bin = ParsedBinFileHandler().load(self.in_dataset_path)
		out_dataset = self._sents_to_extractions(in_parsed_bin.get_parsed_texts(), extracted_sentences, extracted_predicates)
		ExtractedFileHandler(in_parsed_bin.parser).save(out_dataset_path, multi_word_extractions + out_dataset)

	def create_dataset(self, out_dataset_path: str):
		in_parsed_bin = ParsedBinFileHandler().load(self.in_dataset_path)
		out_dataset = self._sents_to_extractions(in_parsed_bin.get_parsed_texts(), None, None)
		ExtractedFileHandler(in_parsed_bin.parser).save(out_dataset_path, out_dataset)
