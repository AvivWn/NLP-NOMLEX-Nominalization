from dataclasses import dataclass, field
from typing import Union, List, Optional

from yet_another_verb.data_handling.bytes.compressed.compressed_parsed_text import CompressedParsedText
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.dependency_parsing.dependency_parser.parsed_word import ParsedWord
from yet_another_verb.dependency_parsing.dependency_parser.parsed_span import ParsedSpan

Words = Union[List[str], ParsedText, CompressedParsedText, 'TokenizedText']


@dataclass
class TokenizedText:
	text_words: List[str]
	words: Optional[Words] = field(default=None)

	def __len__(self) -> int:
		return len(self.text_words)

	def __getitem__(self, i) -> Union[str, ParsedWord, ParsedSpan]:
		try:
			return self.words[i]
		except TypeError:
			return self.text_words[i]

	@property
	def text(self):
		return " ".join(self.text_words)
