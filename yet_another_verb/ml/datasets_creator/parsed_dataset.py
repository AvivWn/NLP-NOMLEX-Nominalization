from os.path import isfile

from tqdm import tqdm

from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.dependency_parser.parsed_bin import ParsedBin
from yet_another_verb.file_handlers import ParsedBinFileHandler
from yet_another_verb.file_handlers.txt_file_handler import TXTFileHandler
from yet_another_verb.utils.linguistic_utils import clean_sentence, is_english
from yet_another_verb.ml.datasets_creator.dataset_creator import DatasetCreator

SENT_MIN_LEN = 3
SENT_MAX_LEN = 35


class ParsedDatasetCreator(DatasetCreator):
	def __init__(self, in_dataset_path: str, dependency_parser: DependencyParser, dataset_size=None, **kwargs):
		super().__init__(dataset_size)
		self.in_dataset_path = in_dataset_path
		self.dependency_parser = dependency_parser
		self.parsed_bin_file_handler = ParsedBinFileHandler(self.dependency_parser)

	@staticmethod
	def _should_use_sentence(sent) -> bool:
		n_words = len(sent.split(" "))

		if n_words >= SENT_MAX_LEN or n_words <= SENT_MIN_LEN:
			return False

		if not is_english(sent):
			return False

		return True

	def _parse_dataset(self, sents) -> ParsedBin:
		parsed_bin = self.dependency_parser.generate_parsed_bin()

		for sent in tqdm(sents):
			if not self._should_use_sentence(sent):
				continue

			sent = clean_sentence(sent)
			doc = self.dependency_parser.parse(sent)

			# Ignore parsings that contain more than one sentence
			if len(list(doc.sents)) > 1:
				continue

			parsed_bin.add(doc)

			if self.has_reached_size(parsed_bin):
				break

		return parsed_bin

	def is_dataset_exist(self, out_dataset_path) -> bool:
		dependency_related_path = self.parsed_bin_file_handler.extend_file_name(out_dataset_path)

		return isfile(dependency_related_path)

	def create_dataset(self, out_dataset_path):
		in_dataset = TXTFileHandler().load(self.in_dataset_path)
		out_dataset = self._parse_dataset(in_dataset)
		self.parsed_bin_file_handler.save(out_dataset_path, out_dataset)
