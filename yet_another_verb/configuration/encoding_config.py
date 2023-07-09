from enum import Enum

from yet_another_verb.sentence_encoding.argument_encoding.full_arg_encoder import FullTextArgumentEncoder
from yet_another_verb.sentence_encoding.argument_encoding.head_idx_arg_encoder import \
	HeadIdxSentenceContextArgumentEncoder, HeadIdxArgContextArgumentEncoder, HeadIdxNoContextArgumentEncoder

from yet_another_verb.sentence_encoding.torch_pretrained_encoder import TorchPretrainedEncoder
from yet_another_verb.sentence_encoding.fasttext_encoder import FastTextEncoder


class EncodingFramework(str, Enum):
	PRETRAINED_TORCH = "pretrained_torch"
	FASTTEXT = "fasttext"


class EncodingLevel(str, Enum):
	HEAD_IDX_IN_SENTENCE_CONTEXT = "head-idx-in-sentence-context"
	HEAD_IDX_IN_ARG_CONTEXT = "head-idx-in-arg-context"
	HEAD_IDX_NO_CONTEXT = "head-idx-no-context"
	FULL_TEXT = "full-text"


class EncodingConfig:
	def __init__(
			self,
			encoding_framework=EncodingFramework.PRETRAINED_TORCH,
			encoder_name="bert-large-uncased",  # "roberta-large",
			device=None,
			encoding_level=EncodingLevel.HEAD_IDX_IN_ARG_CONTEXT
	):
		self.ENCODING_FRAMEWORK = encoding_framework
		self.ENCODER_NAME = encoder_name
		self.DEVICE = device

		self.ENCODING_LEVEL = encoding_level


ENCODING_CONFIG = EncodingConfig()

ENCODER_BY_FRAMEWORK = {
	EncodingFramework.PRETRAINED_TORCH: TorchPretrainedEncoder,
	EncodingFramework.FASTTEXT: FastTextEncoder
}
FRAMEWORK_BY_ENCODER = {v: k for k, v in ENCODER_BY_FRAMEWORK.items()}

ARG_ENCODER_BY_LEVEL = {
	EncodingLevel.HEAD_IDX_IN_SENTENCE_CONTEXT: HeadIdxSentenceContextArgumentEncoder,
	EncodingLevel.HEAD_IDX_IN_ARG_CONTEXT: HeadIdxArgContextArgumentEncoder,
	EncodingLevel.HEAD_IDX_NO_CONTEXT: HeadIdxNoContextArgumentEncoder,
	EncodingLevel.FULL_TEXT: FullTextArgumentEncoder
}
ARG_LEVEL_BY_ARG_ENCODER = {v: k for k, v in ARG_ENCODER_BY_LEVEL.items()}
