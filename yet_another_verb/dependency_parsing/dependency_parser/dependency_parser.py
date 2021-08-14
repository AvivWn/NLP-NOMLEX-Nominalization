import abc

from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.dependency_parsing.dependency_parser.input_text import InputText


class DependencyParser(abc.ABC):
	@abc.abstractmethod
	def __call__(self, text: InputText): pass

	@abc.abstractmethod
	def parse(self, text: InputText) -> ParsedText: pass
