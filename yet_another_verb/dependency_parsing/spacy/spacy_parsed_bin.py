from typing import Iterator, List

from spacy.tokens import DocBin

from yet_another_verb.dependency_parsing.dependency_parser.parsed_bin import ParsedBin
from yet_another_verb.dependency_parsing.spacy.spacy_parsed_text import SpacyParsedText
from yet_another_verb.dependency_parsing.spacy.spacy_parser import SpacyParser

SAVED_ATTRS = ["ID", "ORTH", "LEMMA", "TAG", "POS", "HEAD", "DEP", "ENT_IOB", "ENT_TYPE"]


class SpacyParsedBin(ParsedBin):
	_doc_bin: DocBin

	def __init__(self, parser: SpacyParser):
		super().__init__(parser)
		self._doc_bin = DocBin(attrs=SAVED_ATTRS, store_user_data=True)

	def __len__(self) -> int:
		return len(self._doc_bin)

	def add(self, parsed_text: SpacyParsedText):
		self._doc_bin.add(parsed_text.get_inner())

	def add_multiple(self, parsed_texts: List[SpacyParsedText]):
		for parsed_text in parsed_texts:
			self.add(parsed_text)

	def get_parsed_texts(self) -> Iterator[SpacyParsedText]:
		return map(SpacyParsedText, self._doc_bin.get_docs(self.parser.vocab))

	def to_bytes(self) -> bytes:
		return self._doc_bin.to_bytes()

	def from_bytes(self, bytes_data: bytes):
		self._doc_bin.from_bytes(bytes_data)
