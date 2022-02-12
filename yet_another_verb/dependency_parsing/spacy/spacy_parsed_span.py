from typing import Union, Iterator, TYPE_CHECKING

from spacy.tokens import Token, Span

from yet_another_verb.dependency_parsing.dependency_parser.parsed_span import ParsedSpan
from yet_another_verb.dependency_parsing.spacy.spacy_parsed_word import SpacyParsedWord

if TYPE_CHECKING:
	from yet_another_verb.dependency_parsing.spacy.spacy_parsed_text import SpacyParsedText


class SpacyParsedSpan(ParsedSpan):
	_span: Span

	def __init__(self, span: Span):
		super().__init__()
		self._span = span

	def __len__(self) -> int:
		return len(self._span)

	def __getitem__(self, i) -> Union[SpacyParsedWord, 'SpacyParsedSpan']:
		item = self._span[i]

		if isinstance(item, Token):
			return SpacyParsedWord(item)
		else:  # span
			return SpacyParsedSpan(item)

	def __str__(self) -> str:
		return str(self._span)

	def __repr__(self) -> str:
		return repr(self._span)

	def __iter__(self) -> Iterator[SpacyParsedWord]:
		return map(SpacyParsedWord, iter(self._span))

	@property
	def root(self) -> SpacyParsedWord:
		return SpacyParsedWord(self._span.root)

	@property
	def text(self) -> str:
		return self._span.text

	def as_standalone_parsed_text(self) -> 'SpacyParsedText':
		from yet_another_verb.dependency_parsing.spacy.spacy_parsed_text import SpacyParsedText
		return SpacyParsedText(self._span.as_doc())
