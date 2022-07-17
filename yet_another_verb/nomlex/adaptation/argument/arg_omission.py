from yet_another_verb.arguments_extractor.extraction.argument.argument_type import ARGUMENT_TYPES


def is_supported_arg_type(arg_type: str) -> bool:
	return arg_type in ARGUMENT_TYPES


def should_ommit_argument(arg_type: str) -> bool:
	return not is_supported_arg_type(arg_type)
