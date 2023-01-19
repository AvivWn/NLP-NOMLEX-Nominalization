from yet_another_verb.sentence_encoding.torch_pretrained_encoder import TorchPretrainedEncoder
from yet_another_verb.sentence_encoding.fasttext_encoder import FastTextEncoder

PRETRAINED_TORCH_FRAMEWORK = "pretrained_torch"
FASTTEXT_FRAMEWORK = "fasttext"


encoder_by_framework = {
	PRETRAINED_TORCH_FRAMEWORK: TorchPretrainedEncoder,
	FASTTEXT_FRAMEWORK: FastTextEncoder
}
framework_by_encoder = {v: k for k, v in encoder_by_framework.items()}
