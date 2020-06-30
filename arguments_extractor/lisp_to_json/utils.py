from collections import defaultdict
from copy import deepcopy

# Needed for knowing the "known" constants
from arguments_extractor.constants.ud_constants import *
from arguments_extractor.constants.lexicon_constants import *

# For debug
unknown_values_dict = defaultdict(set)
known_values_dict = defaultdict(set)

# For exceptions and errors
curr_specs = {"word": None, "subcat": None, "comp": None, "is_verb": None}

# Some useful functions for translating from Lisp to Json

def is_known(value, types, type_for_unknown):
	for constant_name, constant_value in globals().items():
		if constant_value == value and any([constant_name.startswith(type_of_constant) for type_of_constant in types]):
			known_values_dict[type_for_unknown].add(value)
			return True

	unknown_values_dict[type_for_unknown].add(value)

	return False

def get_current_specs():
	curr_specs_list = []

	for position_key, position_value in curr_specs.items():
		if position_value is not None:
			curr_specs_list.append(f"{position_key}: {position_value}")

	return ", ".join(curr_specs_list)

def without_part(a_type):
	a_type = a_type.replace("VERB-PART", "VERB-NOM").replace("-PART-", "-")

	if a_type.startswith("NOM"):
		return a_type.replace("-PART", "-INTRANS")

	return a_type.replace("-PART", "")

def get_right_value(table, subcat_type, default=None, is_verb=False):
	if subcat_type not in table.keys():
		new_subcat_type = without_part(subcat_type)

		if new_subcat_type != subcat_type:
			return get_right_value(table, new_subcat_type, default=default, is_verb=is_verb)

		return default

	if is_verb:
		return deepcopy(table[subcat_type][0])

	return deepcopy(table[subcat_type][1])