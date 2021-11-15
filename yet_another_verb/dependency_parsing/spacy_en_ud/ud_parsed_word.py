from typing import Iterator, List
from functools import lru_cache

from spacy.tokens import Token

from yet_another_verb.dependency_parsing.dependency_parser.parsed_word import ParsedWord


class UDParsedWord(ParsedWord):
	_token: Token

	def __init__(self, token: Token):
		self._token = token

	def __len__(self) -> int:
		return len(self._token)

	def __str__(self) -> str:
		return str(self._token)

	def __repr__(self) -> str:
		return repr(self._token)

	def __unicode__(self) -> str:
		return self._token.__unicode__()

	def __bytes__(self) -> bytes:
		return self._token.__bytes__()

	def __hash__(self) -> int:
		return hash(self._token)

	def __eq__(self, other: 'UDParsedWord') -> bool:
		return self._token == other._token

	@property
	def i(self) -> int:
		return self._token.i

	@property
	def subtree(self) -> Iterator['UDParsedWord']:
		return map(UDParsedWord, self._token.subtree)

	@property
	@lru_cache(maxsize=None)
	def children(self) -> List['UDParsedWord']:
		return list(map(UDParsedWord, self._token.children))

	@property
	def head(self) -> 'UDParsedWord':
		return UDParsedWord(self._token.head)

	@property
	def dep(self) -> str:
		return self._token.dep_

	@property
	def text(self) -> str:
		return self._token.text

	@property
	def lemma(self) -> str:
		return self._token.lemma_

	@property
	def tag(self) -> str:
		return self._token.tag_

	@property
	def pos(self) -> str:
		return self._token.pos_
