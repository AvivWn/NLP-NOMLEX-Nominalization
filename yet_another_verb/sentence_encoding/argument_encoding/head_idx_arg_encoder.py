from yet_another_verb.arguments_extractor.extraction import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.utils.parsed_argument import get_argument_in_parsed_text
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.sentence_encoding.argument_encoding.arg_encoder import ArgumentEncoder
from yet_another_verb.sentence_encoding.encoder import Encoder


class HeadIdxArgumentEncoder(ArgumentEncoder):
	def __init__(self, parsed_text: ParsedText, encoder: Encoder, context_sentence=True, context_arg=True):
		super().__init__(parsed_text)
		self.encoder = encoder
		self.context_sentence = context_sentence
		self.context_arg = context_arg
		assert not (context_sentence and not context_arg), \
			"Can not encode argument with sentence context and without argument context"

	def encode(self, argument: ExtractedArgument):
		arg_span = get_argument_in_parsed_text(argument, self.parsed_text)
		head_idx = arg_span.root.i

		context = self.parsed_text.tokenized_text
		if not self.context_arg:  # only word context
			context = context.split()[head_idx]
			head_idx = 0
		elif not self.context_sentence:  # only argument context
			context = context.split()[arg_span[0].i: arg_span[-1].i + 1]
			head_idx -= arg_span[0].i

		return self.encoder.encode_word_in_context(context, head_idx).clone()
