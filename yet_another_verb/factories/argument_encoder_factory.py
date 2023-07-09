from argparse import ArgumentParser
from typing import Optional

from yet_another_verb.factories.encoder_factory import EncoderFactory
from yet_another_verb.factories.factory import Factory
from yet_another_verb.sentence_encoding.argument_encoding.arg_encoder import ArgumentEncoder
from yet_another_verb.configuration.encoding_config import ARG_ENCODER_BY_LEVEL, ENCODING_CONFIG


class ArgumentEncoderFactory(Factory):
	def __init__(self, encoding_level: str, **kwargs):
		self.encoding_level = encoding_level
		self.params = kwargs

	def __call__(self) -> ArgumentEncoder:
		encoder = EncoderFactory(**self.params)()
		return ARG_ENCODER_BY_LEVEL[self.encoding_level](**self.params, encoder=encoder)

	@staticmethod
	def expand_parser(arg_parser: Optional[ArgumentParser] = None) -> ArgumentParser:
		arg_parser.add_argument(
			'--encoding-level', default=ENCODING_CONFIG.ENCODING_LEVEL,
			help="The encoding level of an argument"
		)
		arg_parser = EncoderFactory.expand_parser(arg_parser)
		return arg_parser
