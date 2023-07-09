from typing import List

from yet_another_verb.arguments_extractor.extraction import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.utils.argument_utils import get_argument_text
from yet_another_verb.sentence_encoding.argument_encoding.arg_encoder import ArgumentEncoder
from yet_another_verb.sentence_encoding.encoder import Encoder


class FullTextArgumentEncoder(ArgumentEncoder):
	def __init__(self, encoder: Encoder):
		super().__init__(encoder)

	def encode(self, words: List[str], argument: ExtractedArgument):
		arg_text = get_argument_text(words, argument)
		return self.encoder.encode(arg_text).clone()
