import abc
from typing import Iterator

from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText


class ParsedBin(abc.ABC):
	@abc.abstractmethod
	def __len__(self) -> int: pass

	@abc.abstractmethod
	def add(self, parsed_text: ParsedText): pass

	@abc.abstractmethod
	def get_docs(self, parser: DependencyParser) -> Iterator[ParsedText]: pass

	@abc.abstractmethod
	def to_bytes(self) -> bytes: pass

	@staticmethod
	@abc.abstractmethod
	def from_bytes(bytes_data) -> 'ParsedBin': pass
