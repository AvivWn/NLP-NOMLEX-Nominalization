from typing import List

from yet_another_verb.arguments_extractor.extraction import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.utils.argument_utils import get_argument_head_idx
from yet_another_verb.sentence_encoding.argument_encoding.arg_encoder import ArgumentEncoder
from yet_another_verb.sentence_encoding.encoder import Encoder


class HeadIdxArgumentEncoder(ArgumentEncoder):
	def __init__(self, encoder: Encoder, context_sentence=True, context_arg=True, **kwargs):
		super().__init__(encoder, **kwargs)
		self.context_sentence = context_sentence
		self.context_arg = context_arg
		assert not (context_sentence and not context_arg), \
			"Can not encode argument with sentence context and without argument context"

	def encode(self, words: List[str], argument: ExtractedArgument):
		head_idx = get_argument_head_idx(words, argument)

		context = " ".join([str(word) for word in words])
		if not self.context_arg:  # only word context
			context = words[head_idx]
			head_idx = 0
		elif not self.context_sentence:  # only argument context
			assert argument.start_idx >= 0 and argument.end_idx >= 0
			context = " ".join(words[argument.start_idx: argument.end_idx + 1])
			head_idx -= argument.start_idx

		return self.encoder.encode_word_in_context(context, head_idx).clone()


class HeadIdxSentenceContextArgumentEncoder(HeadIdxArgumentEncoder):
	def __init__(self, encoder: Encoder, **kwargs):
		super().__init__(encoder, context_sentence=True, context_arg=True, **kwargs)


class HeadIdxArgContextArgumentEncoder(HeadIdxArgumentEncoder):
	def __init__(self, encoder: Encoder, **kwargs):
		super().__init__(encoder, context_sentence=False, context_arg=True, **kwargs)


class HeadIdxNoContextArgumentEncoder(HeadIdxArgumentEncoder):
	def __init__(self, encoder: Encoder, **kwargs):
		super().__init__(encoder, context_sentence=False, context_arg=False, **kwargs)
