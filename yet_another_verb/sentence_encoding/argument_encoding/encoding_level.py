from enum import Enum

from yet_another_verb.sentence_encoding.argument_encoding.head_idx_arg_encoder import HeadIdxArgumentEncoder


class EncodingLevel(str, Enum):
	HEAD_IDX = "head-idx"


encoder_by_level = {
	EncodingLevel.HEAD_IDX: HeadIdxArgumentEncoder
}
