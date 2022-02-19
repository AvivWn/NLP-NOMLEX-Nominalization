import abc
from collections import Counter
from typing import List, Optional, TypeVar

from yet_another_verb.arguments_extractor.extraction.extraction import Extraction, Extractions
from yet_another_verb.arguments_extractor.extraction.extracted_argument import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.multi_word_extraction import MultiWordExtraction
from yet_another_verb.nomlex.constants.argument_type import ArgumentType

ArgumentTypes = Optional[List[ArgumentType]]
ExtractionRepr = TypeVar("ExtractionRepr")


class ExtractionRepresentation(abc.ABC):
	@abc.abstractmethod
	def _represent_predicate(self, words: list, predicate_idx: int):
		raise NotImplementedError()

	@abc.abstractmethod
	def _represent_argument(self, words: list, predicate_idx: int, argument: ExtractedArgument):
		raise NotImplementedError()

	def _represent_extraction(self, extraction: Extraction, arg_types: ArgumentTypes = None) -> ExtractionRepr:
		extraction_repr = {}

		for arg in extraction.args:
			if arg_types is None or arg.arg_type in arg_types:
				extraction_repr[arg.arg_tag] = self._represent_argument(extraction.words, extraction.predicate_idx, arg)

		return extraction_repr

	def _represent_list(self, extractions: Extractions, arg_types: ArgumentTypes = None) -> List[ExtractionRepr]:
		list_repr = [self._represent_extraction(e, arg_types) for e in extractions]
		return list_repr

	def _represent_combined_list(
			self, extractions: Extractions, safe_combine=False, arg_types: ArgumentTypes = None
	) -> Optional[ExtractionRepr]:
		if len(extractions) == 0:
			raise Exception("A predicate with no extractions cannot be represented as combined.")

		ex = extractions[0]
		words, predicate_idx, predicate_lemma = ex.words, ex.predicate_idx, ex.predicate_lemma

		args, arg_tag_counts, arg_idx_counts = set(), Counter(), Counter()
		for extraction in extractions:
			args.update(extraction.args)
			arg_tag_counts.update(extraction.arg_types)
			arg_idx_counts.update(extraction.arg_indices)

		if safe_combine:
			limited_args = []
			for arg in args:
				if arg_tag_counts[arg.arg_tag] <= 1 and all([arg_idx_counts[i] <= 1 for i in arg.arg_idxs]):
					limited_args.append(arg)

			args = limited_args

		combined_extraction = Extraction(words=words, predicate_idx=predicate_idx, predicate_lemma=predicate_lemma, args=args)
		return self._represent_extraction(combined_extraction, arg_types)

	def represent_dict(self, multi_word_ext: MultiWordExtraction, arg_types: ArgumentTypes = None) -> dict:
		dict_repr = {}

		for predicate_idx, extractions in multi_word_ext.extractions_per_idx.items():
			extractions_repr = self._represent_list(extractions, arg_types)
			dict_repr[self._represent_predicate(multi_word_ext.words, predicate_idx)] = extractions_repr

		return dict_repr

	def represent_combined_dict(
			self, multi_word_ext: MultiWordExtraction,
			safe_combine=False, arg_types: ArgumentTypes = None
	) -> dict:
		dict_repr = {}

		for predicate_idx, extractions in multi_word_ext.extractions_per_idx.items():
			if len(extractions) != 0:
				extractions_repr = self._represent_combined_list(extractions, safe_combine, arg_types)
				dict_repr[self._represent_predicate(multi_word_ext.words, predicate_idx)] = extractions_repr

		return dict_repr
