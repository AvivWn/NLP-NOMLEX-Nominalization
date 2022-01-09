from typing import Iterator

from spacy.tokens import DocBin

from yet_another_verb.dependency_parsing.dependency_parser.parsed_bin import ParsedBin
from yet_another_verb.dependency_parsing.spacy.spacy_parsed_text import SpacyParsedText
from yet_another_verb.dependency_parsing.spacy.spacy_parser import UDParser


class SpacyParsedBin(ParsedBin):
	_doc_bin: DocBin

	def __init__(self, attrs=None, store_user_data=False):
		self._doc_bin = DocBin(attrs=attrs, store_user_data=store_user_data)

	def __len__(self) -> int:
		return len(self._doc_bin)

	def add(self, parsed_text: SpacyParsedText):
		self._doc_bin.add(parsed_text.get_inner())

	def get_parsed_texts(self, parser: UDParser) -> Iterator[SpacyParsedText]:
		return map(SpacyParsedText, self._doc_bin.get_docs(parser.vocab))

	def to_bytes(self) -> bytes:
		return self._doc_bin.to_bytes()

	@staticmethod
	def from_bytes(bytes_data) -> 'SpacyParsedBin':
		parsed_bin = SpacyParsedBin(attrs=[])
		parsed_bin._doc_bin = DocBin().from_bytes(bytes_data)
		return parsed_bin
