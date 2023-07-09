import abc
from typing import List, Optional, TypeVar, Union

from yet_another_verb.arguments_extractor.extraction.utils.combination import combine_extractions
from yet_another_verb.arguments_extractor.extraction import Extraction, Extractions, ExtractedArgument, \
	MultiWordExtraction
from yet_another_verb.arguments_extractor.extraction.argument.argument_type import ArgumentTypes

ExtractionRepr = TypeVar("ExtractionRepr")


class ExtractionRepresentation(abc.ABC):
	def __init__(self, arg_types: Optional[ArgumentTypes] = None):
		self.arg_types = arg_types

	@abc.abstractmethod
	def _represent_predicate(self, words: list, predicate_idx: int):
		raise NotImplementedError()

	@abc.abstractmethod
	def _represent_argument(self, words: list, predicate_idx: int, argument: ExtractedArgument):
		raise NotImplementedError()

	def represent_single(self, extraction: Extraction) -> ExtractionRepr:
		extraction_repr = {}

		for arg in extraction.args:
			if self.arg_types is None or arg.arg_type in self.arg_types:
				extraction_repr[arg.arg_tag] = self._represent_argument(extraction.words, extraction.predicate_idx, arg)

		return extraction_repr

	def represent_multiple(self, extractions: Extractions, combined=False) \
		-> Optional[Union[ExtractionRepr, List[ExtractionRepr]]]:
		if combined:
			extraction = combine_extractions(extractions, safe_combine=True)
			if len(extraction.args) == 0:
				return None

			return self.represent_single(extraction)

		list_repr = [self.represent_single(e) for e in extractions]
		return list_repr

	def represent_by_word(self, multi_word_ext: MultiWordExtraction, combined=False) -> dict:
		dict_repr = {}
		for predicate_idx, extractions in multi_word_ext.extractions_per_idx.items():
			if len(extractions) != 0:
				extractions_repr = self.represent_multiple(extractions, combined=combined)

				if extractions_repr is not None:
					dict_repr[self._represent_predicate(multi_word_ext.words, predicate_idx)] = extractions_repr

		return dict_repr
