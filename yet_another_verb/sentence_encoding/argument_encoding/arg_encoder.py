import abc

import torch

from yet_another_verb.arguments_extractor.extraction import ExtractedArgument
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText


class ArgumentEncoder(abc.ABC):
	def __init__(self, words: ParsedText):
		self.words = words

	@abc.abstractmethod
	def encode(self, argument: ExtractedArgument) -> torch.Tensor:
		pass
