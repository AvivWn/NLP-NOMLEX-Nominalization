from enum import Enum
from functools import partial

from yet_another_verb.sentence_encoding.argument_encoding.full_arg_encoder import FullTextArgumentEncoder
from yet_another_verb.sentence_encoding.argument_encoding.head_idx_arg_encoder import HeadIdxArgumentEncoder


class EncodingLevel(str, Enum):
	HEAD_IDX_IN_SENTENCE_CONTEXT = "head-idx-in-sentence-context"
	HEAD_IDX_IN_ARG_CONTEXT = "head-idx-in-arg-context"
	HEAD_IDX_NO_CONTEXT = "head-id-no-context"
	FULL_TEXT = "full-text"


encoder_by_level = {
	"head-idx": partial(HeadIdxArgumentEncoder, context_sentence=True, context_arg=True),  # backward compatability
	EncodingLevel.HEAD_IDX_IN_SENTENCE_CONTEXT: partial(HeadIdxArgumentEncoder, context_sentence=True, context_arg=True),
	EncodingLevel.HEAD_IDX_IN_ARG_CONTEXT: partial(HeadIdxArgumentEncoder, context_sentence=False, context_arg=True),
	EncodingLevel.HEAD_IDX_NO_CONTEXT: partial(HeadIdxArgumentEncoder, context_sentence=False, context_arg=False),
	EncodingLevel.FULL_TEXT: FullTextArgumentEncoder
}
