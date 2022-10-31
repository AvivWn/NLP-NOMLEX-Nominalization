from dataclasses import dataclass, field
from typing import Optional

from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText


@dataclass
class CompressedParsedText:
	bytes_data: bytes
	parsing_egnine: str
	parser_name: str

	parsed_text: Optional[ParsedText] = field(default=None)
