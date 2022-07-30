import abc
from typing import Iterator, Dict

from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText


class ParsedBin(abc.ABC):
	def __init__(self, parser: DependencyParser):
		self.parser = parser

	@abc.abstractmethod
	def __len__(self) -> int: pass

	@abc.abstractmethod
	def add(self, parsed_text: ParsedText): pass

	@abc.abstractmethod
	def get_parsed_texts(self) -> Iterator[ParsedText]: pass

	@abc.abstractmethod
	def to_bytes(self) -> bytes: pass

	@abc.abstractmethod
	def from_bytes(self, bytes_data: bytes): pass

	def get_parsing_by_text(self) -> Dict[str, ParsedText]:
		return {doc.text: doc for doc in self.get_parsed_texts()}
