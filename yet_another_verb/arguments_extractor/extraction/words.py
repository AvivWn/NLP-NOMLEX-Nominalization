from typing import Union, List

from yet_another_verb.data_handling.bytes.compressed.compressed_parsed_text import CompressedParsedText
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText

Words = Union[List[str], ParsedText, CompressedParsedText]
