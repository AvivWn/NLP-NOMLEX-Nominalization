from typing import Optional

from yet_another_verb.arguments_extractor.extraction import Extractions
from yet_another_verb.arguments_extractor.extractors.constraints_based.dep_constraints_args_extractor import \
	DepConstraintsArgsExtractor
from yet_another_verb.configuration.parsing_config import PARSING_CONFIG
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory
from yet_another_verb.nomlex.nomlex_version import NomlexVersion
from yet_another_verb.utils.debug_utils import timeit
from yet_another_verb.utils.print_utils import print_as_title_if_verbose


class NomlexArgsExtractor(DepConstraintsArgsExtractor):
	def __init__(
			self,
			nomlex_version: NomlexVersion,
			dependency_parser: DependencyParser = DependencyParserFactory(parser_name=PARSING_CONFIG.PARSER_NAME)(),
			**kwargs
	):
		super().__init__(dependency_parser, **kwargs)
		from yet_another_verb.nomlex.nomlex_maestro import NomlexMaestro
		self.adapted_lexicon = timeit(NomlexMaestro(nomlex_version).get_adapted_lexicon)()

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

		if allow_related_forms:
			self.adapted_lexicon.enhance_orths(limited_predicates)

		return word.lemma in limited_predicates

	def _is_potential_sentence(
			self, words: list,
			limited_predicates: Optional[list], allow_related_forms: bool
	) -> bool:
		if limited_predicates is None:
			return True

		lemmas = set([w.lemma for w in words])

		if allow_related_forms:
			self.adapted_lexicon.enhance_orths(limited_predicates)

		return not lemmas.isdisjoint(limited_predicates)

	def extract(self, word_idx: int, words: ParsedText) -> Optional[Extractions]:
		word = words[word_idx]
		word_entry = self.adapted_lexicon.entries.get(word.lemma, None)

		if word_entry is None:
			return []

		print_as_title_if_verbose(f"{word.text} ({word.lemma}, {word.pos})")
		relevant_constraints_maps = word_entry.get_relevant_constraints_maps(word=word, order_by_potential=True)
		return self._extract_by_constraints_maps(word_idx, words, relevant_constraints_maps)
