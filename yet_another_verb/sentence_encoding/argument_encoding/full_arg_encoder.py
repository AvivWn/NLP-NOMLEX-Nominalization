from yet_another_verb.arguments_extractor.extraction import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.utils.argument_utils import get_argument_text
from yet_another_verb.arguments_extractor.extraction.words import Words
from yet_another_verb.sentence_encoding.argument_encoding.arg_encoder import ArgumentEncoder
from yet_another_verb.sentence_encoding.encoder import Encoder


class FullTextArgumentEncoder(ArgumentEncoder):
	def __init__(self, words: Words, encoder: Encoder):
		super().__init__(words)
		self.encoder = encoder

	def encode(self, argument: ExtractedArgument):
		arg_text = get_argument_text(self.words, argument)
		return self.encoder.encode(arg_text).clone()
