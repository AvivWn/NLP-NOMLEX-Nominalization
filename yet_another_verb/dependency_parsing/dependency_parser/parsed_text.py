import abc
from typing import Union, Iterator

from yet_another_verb.dependency_parsing.dependency_parser.parsed_word import ParsedWord
from yet_another_verb.dependency_parsing.dependency_parser.parsed_span import ParsedSpan


class ParsedText(abc.ABC):
	@abc.abstractmethod
	def __len__(self) -> int: pass

	@abc.abstractmethod
	def __getitem__(self, i) -> Union[ParsedWord, ParsedSpan]: pass

	@abc.abstractmethod
	def __str__(self) -> str: pass

	@abc.abstractmethod
	def __repr__(self) -> str: pass

	@abc.abstractmethod
	def __unicode__(self) -> str: pass

	@abc.abstractmethod
	def __bytes__(self) -> bytes: pass

	@abc.abstractmethod
	def __iter__(self) -> Iterator[ParsedWord]: pass

	@property
	@abc.abstractmethod
	def sents(self) -> Iterator[ParsedSpan]: pass

	@property
	@abc.abstractmethod
	def text(self) -> str: pass

	@property
	def tokenized_text(self) -> str:
		return " ".join([w.text for w in self])