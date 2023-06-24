from os.path import join, isfile, splitext
from typing import List

from yet_another_verb.data_handling.db.encoded_extractions.dataset_loader import EncodedExtractionsLoader
from yet_another_verb.data_handling import ExtractedFileHandler
from yet_another_verb.data_handling.dataset_creator import DatasetCreator
from yet_another_verb.data_handling.file.handlers.extracted_file_handler import EXTRACTED_FILE
from yet_another_verb.dependency_parsing import POSTag
from yet_another_verb.sentence_encoding.argument_encoding.encoding_level import EncodingLevel


class ExtractedFromDBDatasetCreator(DatasetCreator):
	def __init__(
			self, in_dataset_path: str,
			parsing_engine: str, parser_name: str, extraction_mode: str,
			encoding_framework: str, encoder_name: str, encoding_level: EncodingLevel,
			limited_postags: List[POSTag],
			dataset_size=None,
			compress_internally=False,  # save binary of parsings and encodings in the extraction binary
			**kwargs
	):
		super().__init__(dataset_size)
		self.in_dataset_path = in_dataset_path
		self.parsing_engine = parsing_engine
		self.parser_name = parser_name
		self.extraction_mode = extraction_mode
		self.encoding_framework = encoding_framework
		self.encoder_name = encoder_name
		self.encoding_level = encoding_level
		self.limited_postags = limited_postags
		self.compress_internally = compress_internally

	def is_dataset_exist(self, out_dataset_path) -> bool:
		return (self.compress_internally and isfile(out_dataset_path)) or \
			(not self.compress_internally and isfile(join(splitext(out_dataset_path)[0], EXTRACTED_FILE)))

	def append_dataset(self, out_dataset_path: str):
		raise NotImplementedError

	def create_dataset(self, out_dataset_path: str):
		with EncodedExtractionsLoader(
				self.in_dataset_path, self.parsing_engine, self.parser_name, keep_compressed=self.compress_internally) as loader:
			extractions = loader.get_encoded_extractions(
				self.extraction_mode, self.encoding_framework, self.encoder_name,
				EncodingLevel.HEAD_IDX_IN_SENTENCE_CONTEXT, self.limited_postags)

		ExtractedFileHandler(loader.parser, compress_internally=self.compress_internally).save(out_dataset_path, extractions)
