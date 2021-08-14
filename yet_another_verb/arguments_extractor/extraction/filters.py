from typing import *

from yet_another_verb.arguments_extractor.extraction.extraction import Extraction


def choose_longest(extractions: List[Extraction]) -> List[Extraction]:
	extractions.sort(key=lambda e: len(e), reverse=True)
	extractions = list(filter(lambda e: len(e) == len(extractions[0]), extractions))
	return extractions


def uniqify(extractions: List[Extraction]) -> List[Extraction]:
	unique_extractions = []

	for e in extractions:
		if e in unique_extractions:
			continue

		unique_extractions.append(e)

	return unique_extractions
