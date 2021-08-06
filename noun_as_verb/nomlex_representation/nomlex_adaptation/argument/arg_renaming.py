from typing import Union

from noun_as_verb.nomlex_representation.lexicon_constants import SubcatProperty, ArgumentType


def _update_requirements(old_type: Union[str, ArgumentType], new_type: Union[str, ArgumentType, None], subcat: dict):
	requires = subcat.get(SubcatProperty.REQUIRED, {})
	optionals = subcat.get(SubcatProperty.OPTIONAL, {})

	# Update requirements only if specified for the original type
	if old_type not in requires and old_type not in optionals:
		return

	info_to_udpate = requires if old_type in requires else optionals

	if new_type is None:
		info_to_udpate.remove(old_type)
		return

	if isinstance(info_to_udpate, dict):
		info_to_udpate[new_type] = info_to_udpate.pop(old_type)
	elif isinstance(info_to_udpate, list):
		info_to_udpate.remove(old_type)
		info_to_udpate.append(new_type)


def rename_argument(old_type: Union[str, ArgumentType], new_type: Union[str, ArgumentType, None], subcat: dict):
	if old_type == new_type:
		return

	old_arg = subcat.pop(old_type, None)
	if old_arg is None:
		return

	if new_type is not None:
		subcat[new_type] = old_arg

	_update_requirements(old_type, new_type, subcat)
