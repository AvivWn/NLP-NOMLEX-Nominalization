from yet_another_verb.sentence_encoding.torch_pretrained_encoder import TorchPretrainedEncoder

PRETRAINED_TORCH_FRAMEWORK = "pretrained_torch"


encoder_by_framework = {
	PRETRAINED_TORCH_FRAMEWORK: TorchPretrainedEncoder
}
framework_by_encoder = {v: k for k, v in encoder_by_framework.items()}
