from yet_another_verb.arguments_extractor.extraction import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.utils.parsed_argument import get_argument_in_parsed_text
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.sentence_encoding.argument_encoding.arg_encoder import ArgumentEncoder
from yet_another_verb.sentence_encoding.encoder import Encoder


class FullTextArgumentEncoder(ArgumentEncoder):
	def __init__(self, parsed_text: ParsedText, encoder: Encoder):
		super().__init__(parsed_text)
		self.encoder = encoder

	def encode(self, argument: ExtractedArgument):
		arg_span = get_argument_in_parsed_text(argument, self.parsed_text)
		return self.encoder.encode(arg_span.text).clone()
