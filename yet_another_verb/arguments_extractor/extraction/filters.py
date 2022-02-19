from typing import List, Callable, Any
from copy import deepcopy

from yet_another_verb.arguments_extractor.extraction.extraction import Extraction, Extractions


def _choose_first_by_order(extractions: Extractions, order: Callable[[Extraction], Any], reverse: bool) -> Extractions:
	extractions.sort(key=lambda e: order(e), reverse=reverse)
	filtered_extractions = filter(lambda e: order(e) == order(extractions[0]), extractions)
	return list(filtered_extractions)


def prefer_by_n_args(extractions: Extractions) -> Extractions:
	return _choose_first_by_order(extractions, lambda e: len(e), reverse=True)


def prefer_by_constraints(extractions: Extractions) -> Extractions:
	return _choose_first_by_order(extractions, lambda e: len(e.fulfilled_constraints), reverse=True)


def uniqify(extractions: Extractions) -> Extractions:
	unique_extractions = []

	for e in extractions:
		already_found_extraction = False
		for chose_e in unique_extractions:
			if e == chose_e:
				already_found_extraction = True
				break

		if not already_found_extraction:
			unique_extractions.append(e)

	return unique_extractions
