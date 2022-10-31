import abc

import torch

from yet_another_verb.arguments_extractor.extraction import ExtractedArgument
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText


class ArgumentEncoder(abc.ABC):
	def __init__(self, parsed_text: ParsedText):
		self.parsed_text = parsed_text

	@abc.abstractmethod
	def encode(self, argument: ExtractedArgument) -> torch.Tensor:
		pass
