from typing import Union, Iterator

from spacy.tokens import Doc, Token

from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.dependency_parsing.spacy_en_ud.ud_parsed_word import UDParsedWord
from yet_another_verb.dependency_parsing.spacy_en_ud.ud_parsed_span import UDParsedSpan


class UDParsedText(ParsedText):
	_doc: Doc

	def __init__(self, doc: Doc):
		self._doc = doc

	def __len__(self) -> int:
		return len(self._doc)

	def __getitem__(self, i) -> Union[UDParsedWord, UDParsedSpan]:
		item = self._doc[i]

		if isinstance(item, Token):
			return UDParsedWord(item)
		else:  # span
			return UDParsedSpan(item)

	def __str__(self) -> str:
		return str(self._doc)

	def __repr__(self) -> str:
		return repr(self._doc)

	def __unicode__(self) -> str:
		return self._doc.__unicode__()

	def __bytes__(self) -> bytes:
		return self._doc.__bytes__()

	def __iter__(self) -> Iterator[UDParsedWord]:
		return map(UDParsedWord, iter(self._doc))

	@property
	def sents(self) -> Iterator[UDParsedSpan]:
		return self._doc.sents

	@property
	def text(self) -> str:
		return self._doc.text

	def get_inner(self) -> Doc:
		return self._doc
