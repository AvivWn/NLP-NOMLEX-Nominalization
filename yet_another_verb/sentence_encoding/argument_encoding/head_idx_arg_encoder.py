from yet_another_verb.arguments_extractor.extraction import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.utils.argument_utils import get_argument_head_idx
from yet_another_verb.arguments_extractor.extraction.words import Words
from yet_another_verb.sentence_encoding.argument_encoding.arg_encoder import ArgumentEncoder
from yet_another_verb.sentence_encoding.encoder import Encoder


class HeadIdxArgumentEncoder(ArgumentEncoder):
	def __init__(self, words: Words, encoder: Encoder, context_sentence=True, context_arg=True):
		super().__init__(words)
		self.encoder = encoder
		self.context_sentence = context_sentence
		self.context_arg = context_arg
		assert not (context_sentence and not context_arg), \
			"Can not encode argument with sentence context and without argument context"

	def encode(self, argument: ExtractedArgument):
		head_idx = get_argument_head_idx(self.words, argument)

		context = " ".join([str(word) for word in self.words])
		if not self.context_arg:  # only word context
			context = self.words[head_idx]
			head_idx = 0
		elif not self.context_sentence:  # only argument context
			assert argument.start_idx >= 0 and argument.end_idx >= 0
			context = self.words[argument.start_idx: argument.end_idx + 1]
			head_idx -= argument.start_idx

		return self.encoder.encode_word_in_context(context, head_idx).clone()
