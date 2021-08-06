from noun_as_verb.nomlex_representation.lexicon_constants.argument_type import ARGUMENT_TYPES


def is_supported_arg_type(arg_type: str) -> bool:
	return arg_type in ARGUMENT_TYPES


def should_ommit_argument(arg_type: str) -> bool:
	return not is_supported_arg_type(arg_type)
