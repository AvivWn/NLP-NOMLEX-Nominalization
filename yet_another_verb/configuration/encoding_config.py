class EncodingConfig:
	def __init__(
			self,
			encoding_framework="pretrained_torch",
			encoder_name="bert-base-uncased",  # "roberta-large",
			device=None
	):
		self.ENCODING_FRAMEWORK = encoding_framework
		self.ENCODER_NAME = encoder_name
		self.DEVICE = device


ENCODING_CONFIG = EncodingConfig()
