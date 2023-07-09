import abc
from typing import Optional

from yet_another_verb.arguments_extractor.args_extractor import ArgsExtractor
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser


class DepRelatedArgsExtractor(ArgsExtractor, abc.ABC):
	def __init__(self, dependency_parser: DependencyParser, **kwargs):
		self.dependency_parser = dependency_parser

	def _tokenize(self, text: str) -> list:
		return self.dependency_parser(text)

	def _is_potential_predicate(
			self, word_idx: int, words: list,
			limited_predicates: Optional[list], limited_postags: Optional[list],
			allow_related_forms: bool
	) -> bool:
		word = words[word_idx]

		if limited_postags is not None:
			if word.tag not in limited_postags and word.pos not in limited_postags:
				return False

		if limited_predicates is None:
			return True

		return word.lemma in limited_predicates

	def _is_potential_sentence(
			self, words: list,
			limited_predicates: Optional[list], allow_related_forms: bool
	) -> bool:
		if limited_predicates is None:
			return True

		lemmas = set([w.lemma for w in words])
		return not lemmas.isdisjoint(limited_predicates)
