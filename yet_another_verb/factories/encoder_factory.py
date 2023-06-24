from argparse import ArgumentParser
from typing import Optional

from yet_another_verb.factories.factory import Factory
from yet_another_verb.sentence_encoding.encoder import Encoder
from yet_another_verb.configuration.encoding_config import ENCODING_CONFIG, ENCODER_BY_FRAMEWORK


class EncoderFactory(Factory):
	def __init__(self, encoding_framework: str = ENCODING_CONFIG.ENCODING_FRAMEWORK, **kwargs):
		self.encoding_framework = encoding_framework
		self.params = kwargs

	def __call__(self) -> Encoder:
		return ENCODER_BY_FRAMEWORK[self.encoding_framework](**self.params)

	@staticmethod
	def expand_parser(arg_parser: Optional[ArgumentParser] = None) -> ArgumentParser:
		arg_parser.add_argument(
			'--encoding-framework', '-f', default=ENCODING_CONFIG.ENCODING_FRAMEWORK,
			help="The framework which will encode the sentences"
		)
		arg_parser.add_argument(
			'--encoder-name', default=ENCODING_CONFIG.ENCODER_NAME,
			help="The name of the encoder"
		)
		arg_parser.add_argument(
			'--encoding-level', default=None,
			help="The encoding level of an argument"
		)
		arg_parser.add_argument(
			'--device', default=ENCODING_CONFIG.DEVICE,
			help="The device which the encoder will run on"
		)
		return arg_parser
