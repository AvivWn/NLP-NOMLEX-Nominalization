from dataclasses import dataclass, field
from typing import Optional, Union

from yet_another_verb.dependency_parsing.dependency_parser.parsed_span import ParsedSpan
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.dependency_parsing.dependency_parser.parsed_word import ParsedWord


@dataclass
class CompressedParsedText:
	bytes_data: bytes
	parsing_egnine: str
	parser_name: str

	parsed_text: Optional[ParsedText] = field(default=None)

	def __len__(self) -> int:
		return len(self.parsed_text)

	def __getitem__(self, i) -> Union[str, ParsedWord, ParsedSpan]:
		return self.parsed_text[i]

	@property
	def text(self):
		return " ".join(self.parsed_text)
