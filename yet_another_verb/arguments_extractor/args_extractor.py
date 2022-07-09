import abc
from typing import Optional, List, Union

from yet_another_verb.arguments_extractor.extraction.comparators.extraction_matcher import ExtractionMatcher
from yet_another_verb.arguments_extractor.extraction.extraction import Extractions
from yet_another_verb.arguments_extractor.extraction.multi_word_extraction import MultiWordExtraction
from yet_another_verb.nomlex.constants import POSTag
from yet_another_verb.nomlex.constants.word_postag import VERB_POSTAGS, NOUN_POSTAGS


class ArgsExtractor(abc.ABC):
	@abc.abstractmethod
	def _tokenize(self, text: str) -> list:
		pass

	@abc.abstractmethod
	def _is_potential_predicate(
			self, word_idx: int, words: list,
			limited_predicates: Optional[list], limited_postags: Optional[list],
			allow_related_forms: bool
	) -> bool:
		pass

	@abc.abstractmethod
	def _is_potential_sentence(
			self, words: list,
			limited_predicates: Optional[list], allow_related_forms: bool
	) -> bool:
		pass

	@abc.abstractmethod
	def extract(self, word_idx: int, words: list) -> Optional[Extractions]:
		pass

	def extract_multiword(
			self, sent: Union[str, list], *,
			limited_idxs: Optional[List[int]] = None,
			limited_predicates: Optional[List[str]] = None,
			limited_postags: Optional[List[POSTag]] = NOUN_POSTAGS + VERB_POSTAGS,
			references: Optional[Extractions] = None,
			reference_matcher: Optional[ExtractionMatcher] = None,
			allow_related_forms: bool = True
	) -> MultiWordExtraction:
		words = sent
		if isinstance(words, str):
			words = self._tokenize(words)

		if references is not None:
			limited_predicates = [] if limited_predicates is None else limited_predicates
			limited_predicates += [e.predicate_lemma for e in references]

		extractions_per_idx = {}

		if not self._is_potential_sentence(words, limited_predicates, allow_related_forms):
			return MultiWordExtraction(words, extractions_per_idx)

		for i in range(len(words)):
			if limited_idxs is not None and i not in limited_idxs:
				continue

			if self._is_potential_predicate(i, words, limited_predicates, limited_postags, allow_related_forms):
				extractions = self.extract(i, words)

				if reference_matcher is not None:
					extractions = reference_matcher.filter_by(extractions, references)

				if len(extractions) > 0:
					extractions_per_idx[i] = extractions

		return MultiWordExtraction(words, extractions_per_idx)
