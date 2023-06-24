from yet_another_verb.sentence_encoding.torch_pretrained_encoder import TorchPretrainedEncoder
from yet_another_verb.sentence_encoding.fasttext_encoder import FastTextEncoder

PRETRAINED_TORCH_FRAMEWORK = "pretrained_torch"
FASTTEXT_FRAMEWORK = "fasttext"


class EncodingConfig:
	def __init__(
			self,
			encoding_framework=PRETRAINED_TORCH_FRAMEWORK,
			encoder_name="bert-base-uncased",  # "roberta-large",
			device=None
	):
		self.ENCODING_FRAMEWORK = encoding_framework
		self.ENCODER_NAME = encoder_name
		self.DEVICE = device


ENCODING_CONFIG = EncodingConfig()

ENCODER_BY_FRAMEWORK = {
	PRETRAINED_TORCH_FRAMEWORK: TorchPretrainedEncoder,
	FASTTEXT_FRAMEWORK: FastTextEncoder
}
FRAMEWORK_BY_ENCODER = {v: k for k, v in ENCODER_BY_FRAMEWORK.items()}
