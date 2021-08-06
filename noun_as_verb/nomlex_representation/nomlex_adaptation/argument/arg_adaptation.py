from typing import Dict, Union, List

from noun_as_verb.nomlex_representation.lexicon_constants import LexiconType, SubcatType, \
	ArgumentType, ArgumentValue
from noun_as_verb.nomlex_representation.nomlex_adaptation.argument.arg_renaming import rename_argument
from noun_as_verb.nomlex_representation.nomlex_adaptation.modifications import get_default_arg_values


def get_adapted_argument(
		subcat_type: SubcatType, arg_type: ArgumentType, arg_values: List[Union[str, ArgumentValue]], lexicon_type: LexiconType
) -> Dict[ArgumentValue, List[str]]:
	leftover_preps = []
	arg_structure = {}

	if ArgumentValue.NONE in arg_values:
		return {}

	for arg_value in arg_values:
		if isinstance(arg_value, ArgumentValue):
			arg_structure[arg_value] = []
		else:
			leftover_preps.append(arg_value)

	relevant_values = get_default_arg_values(subcat_type, arg_type, lexicon_type)
	if len(leftover_preps) > 0 or (not ArgumentType.is_pp_arg(arg_type) and not ArgumentType.is_np_arg(arg_type)):
		for arg_value in relevant_values:
			arg_structure[arg_value] = leftover_preps

	return arg_structure


def adapt_argument(subcat: dict, subcat_type: SubcatType, arg_type: ArgumentType, lexicon_type: LexiconType):
	adapted_arg = get_adapted_argument(subcat_type, arg_type, subcat[arg_type], lexicon_type)

	if len(adapted_arg) > 0:
		subcat[arg_type] = adapted_arg
	else:
		rename_argument(arg_type, None, subcat)
