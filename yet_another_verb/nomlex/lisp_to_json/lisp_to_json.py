from typing import List, Union, Iterator
from itertools import takewhile
from collections import ChainMap

from yet_another_verb.nomlex.lisp_to_json.lisp_units_split import split_lisp_into_units


def _is_tag(value: str) -> bool:
	return value.startswith(":")


def _is_element_start(value: str) -> bool:
	return value == "("


def _is_element_end(value: str) -> bool:
	return value == ")"


def _clean_str_from_quotes(value: str) -> str:
	return value.strip('"')


def _obtain_list(lisp_units: Iterator) -> list:
	obtained_list = []

	for value in takewhile(lambda x: not _is_element_end(x), lisp_units):
		if _is_element_start(value):
			element = _obtain_dictionary(lisp_units)
		else:
			element = _clean_str_from_quotes(value)

		obtained_list.append(element)

	assert all(isinstance(x, type(obtained_list[0])) for x in obtained_list)

	if all(isinstance(x, dict) for x in obtained_list):
		obtained_list = [dict(ChainMap(*obtained_list))]

	return obtained_list


def _obtain_dictionary_value(lisp_units: Iterator) -> Union[str, list]:
	curr_unit = next(lisp_units)

	if _is_element_start(curr_unit):
		obtained_value = _obtain_list(lisp_units)

		if len(obtained_value) == 1 and isinstance(obtained_value[0], dict):
			obtained_value = obtained_value[0]
	else:
		obtained_value = _clean_str_from_quotes(curr_unit)

	return obtained_value


def _obtain_dictionary(lisp_units: Iterator) -> dict:
	dict_data = {}
	dict_type = next(lisp_units)

	for value in takewhile(lambda x: not _is_element_end(x), lisp_units):
		assert _is_tag(value)
		dict_key = value.replace(":", "")
		dict_value = _obtain_dictionary_value(lisp_units)
		dict_data.update({dict_key: dict_value})

	return {dict_type: dict_data}


def lisps_to_jsons(lisp_text: str) -> List[dict]:
	"""
	Reformats a lisp formatted objects to a json formatted objects
	:param lisp_text: lisp formatted objects as string
	:return: json formatted objects
	"""

	lisp_units = iter(split_lisp_into_units(lisp_text))
	json_data = []

	for _ in takewhile(lambda x: _is_element_start(x), lisp_units):
		dictionary = _obtain_dictionary(lisp_units)
		json_data.append(dictionary)

	assert not any(True for _ in lisp_units), "The transformation lisp -> json couldn't use the entire data"  # EOF check
	return json_data
