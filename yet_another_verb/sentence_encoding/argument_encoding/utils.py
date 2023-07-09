from yet_another_verb.configuration.encoding_config import FRAMEWORK_BY_ENCODER, ARG_LEVEL_BY_ARG_ENCODER
from yet_another_verb.sentence_encoding.argument_encoding.arg_encoder import ArgumentEncoder


def arg_encoder_to_tuple_id(arg_encoder: ArgumentEncoder) -> tuple:
	return (
		FRAMEWORK_BY_ENCODER[type(arg_encoder.encoder)],
		arg_encoder.encoder.name,
		ARG_LEVEL_BY_ARG_ENCODER[type(arg_encoder)]
	)
