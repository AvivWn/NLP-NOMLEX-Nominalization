import abc
from typing import Union, Iterator, TYPE_CHECKING

from yet_another_verb.dependency_parsing.dependency_parser.parsed_word import ParsedWord

if TYPE_CHECKING:
	from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText


class ParsedSpan(abc.ABC, list):
	@abc.abstractmethod
	def __len__(self) -> int: pass

	@abc.abstractmethod
	def __getitem__(self, i) -> Union[ParsedWord, 'ParsedSpan']: pass

	@abc.abstractmethod
	def __str__(self) -> str: pass

	@abc.abstractmethod
	def __repr__(self) -> str: pass

	@abc.abstractmethod
	def __iter__(self) -> Iterator[ParsedWord]: pass

	@property
	@abc.abstractmethod
	def root(self) -> ParsedWord: pass

	@property
	@abc.abstractmethod
	def text(self) -> str: pass

	@property
	def tokenized_text(self) -> str:
		return " ".join([w.text for w in self])

	def as_standalone_parsed_text(self) -> 'ParsedText': pass
