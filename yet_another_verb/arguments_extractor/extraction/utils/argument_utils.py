from typing import Union, TYPE_CHECKING

from yet_another_verb.arguments_extractor.extraction import ArgumentType
from yet_another_verb.arguments_extractor.extraction.argument.argument_type import ARGUMENT_TYPE_TO_ACTIVE_VERBAL
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.dependency_parsing.dependency_parser.parsed_span import ParsedSpan
from yet_another_verb.arguments_extractor.extraction.argument.extracted_argument import ExtractedArgument, \
	ExtractedArguments
from yet_another_verb.dependency_parsing.dependency_relation import PREP_RELATIONS, DepRelation

if TYPE_CHECKING:
	from yet_another_verb.arguments_extractor.extraction.words import Words


def get_argument_words(words: 'Words', argument: ExtractedArgument) -> Union['Words', ParsedSpan]:
	return words[argument.start_idx: argument.end_idx + 1]


def get_argument_text(words: 'Words', argument: ExtractedArgument) -> str:
	arg_words = get_argument_words(words, argument)
	return " ".join([str(word) for word in arg_words])


def get_argument_head_idx(words: 'Words', argument: ExtractedArgument) -> int:
	if argument.head_idx is not None:
		return argument.head_idx

	if isinstance(words, ParsedText):
		return get_argument_words(words, argument).root.i

	raise NotImplementedError()


def _get_pp_type_from_text(text: str) -> str:
	from yet_another_verb.configuration.extractors_config import EXTRACTORS_CONFIG

	text = text.lower()
	for pp_value in EXTRACTORS_CONFIG.CANDIDATES_PP_VALUES:
		if text.startswith(pp_value.lower() + " "):
			return pp_value


def specify_pp_type_in_arg(words: ParsedText, arg: ExtractedArgument):
	if arg.arg_type == ArgumentType.PP:
		arg_text = get_argument_text(words, arg).lower()
		pp_type = _get_pp_type_from_text(arg_text)

		if pp_type is None:
			arg_words = get_argument_words(words, arg)
			pp_indices = [child.i for child in arg_words.root.children if child.dep in PREP_RELATIONS]

			if len(pp_indices) > 0:
				text_from_pp = words[pp_indices[0]:].tokenized_text
				pp_type = _get_pp_type_from_text(text_from_pp)

		if pp_type is None:
			pp_type = "OTHER"

		arg.arg_type = f"{ArgumentType.PP}-{pp_type}"


def remove_pp_type_in_arg(arg: ExtractedArgument):
	if arg.arg_type.startswith(ArgumentType.PP):
		arg.arg_type = ArgumentType.PP


def reduce_args_by_arg_types(args: ExtractedArguments, arg_types=None) -> ExtractedArguments:
	relevant_args = set()
	for arg in args:
		if arg_types is not None and arg.arg_type not in arg_types:
			continue

		relevant_args.add(arg)

	return list(relevant_args)


def rename_type_to_verbal_active(arg: ExtractedArgument):
	dep_relation = ARGUMENT_TYPE_TO_ACTIVE_VERBAL.get(arg.arg_type)

	if dep_relation is None:
		for arg_type in ARGUMENT_TYPE_TO_ACTIVE_VERBAL:
			if arg.arg_type.startswith(arg_type):
				dep_relation = arg.arg_type.replace(arg_type, ARGUMENT_TYPE_TO_ACTIVE_VERBAL[arg_type])
				dep_relation = dep_relation.replace("-", ":")

	if dep_relation is not None:
		arg.arg_type = dep_relation
