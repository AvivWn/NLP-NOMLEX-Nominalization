from typing import Union, Iterator

from spacy.tokens import Token, Span

from yet_another_verb.dependency_parsing.dependency_parser.parsed_span import ParsedSpan
from yet_another_verb.dependency_parsing.spacy_en_ud.ud_parsed_word import UDParsedWord


class UDParsedSpan(ParsedSpan):
	_span: Span

	def __init__(self, span: Span):
		self._span = span

	def __len__(self) -> int:
		return len(self._span)

	def __getitem__(self, i) -> Union[UDParsedWord, 'UDParsedSpan']:
		item = self._span[i]

		if isinstance(item, Token):
			return UDParsedWord(item)
		else:  # span
			return UDParsedSpan(item)

	def __str__(self) -> str:
		return str(self._span)

	def __repr__(self) -> str:
		return repr(self._span)

	def __iter__(self) -> Iterator[UDParsedWord]:
		return map(UDParsedWord, iter(self._span))

	@property
	def root(self) -> UDParsedWord:
		return UDParsedWord(self._span.root)

	@property
	def text(self) -> str:
		return self._span.text
