from typing import Union, List

from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.dependency_parsing.dependency_parser.parsed_word import ParsedWord

InputText = Union[str, List[str], ParsedText, List[ParsedWord]]
