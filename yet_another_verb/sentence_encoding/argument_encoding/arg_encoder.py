import abc
from typing import List

import torch

from yet_another_verb.arguments_extractor.extraction import ExtractedArgument
from yet_another_verb.sentence_encoding.encoder import Encoder


class ArgumentEncoder(abc.ABC):
	def __init__(self, encoder: Encoder, **kwargs):
		self.encoder = encoder

	@abc.abstractmethod
	def encode(self, words: List[str], argument: ExtractedArgument) -> torch.Tensor:
		pass
