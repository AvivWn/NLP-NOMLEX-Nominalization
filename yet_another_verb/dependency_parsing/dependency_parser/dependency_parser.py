import abc
from typing import TYPE_CHECKING

from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.dependency_parsing.dependency_parser.input_text import InputText

if TYPE_CHECKING:
	from yet_another_verb.dependency_parsing.dependency_parser.parsed_bin import ParsedBin


class DependencyParser(abc.ABC):
	@abc.abstractmethod
	def __call__(self, text: InputText): pass

	@property
	@abc.abstractmethod
	def name(self) -> str: pass

	@abc.abstractmethod
	def parse(self, text: InputText) -> ParsedText: pass

	def generate_parsed_bin(self) -> 'ParsedBin': pass
