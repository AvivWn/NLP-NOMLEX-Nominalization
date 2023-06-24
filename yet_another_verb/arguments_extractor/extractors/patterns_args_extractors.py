from typing import Optional

from yet_another_verb.arguments_extractor.extraction import Extractions, ArgumentType
from yet_another_verb.arguments_extractor.extractors.dep_constraints_args_extractor import \
	DepConstraintsArgsExtractor
from yet_another_verb.dependency_parsing import DepRelation
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.nomlex.representation.constraints_map import ORConstraintsMaps


class PatternsArgsExtractor(DepConstraintsArgsExtractor):
	def __init__(
			self, dependency_parser: DependencyParser,
			constraints_maps: ORConstraintsMaps,
			**kwargs):
		super().__init__(dependency_parser, **kwargs)
		self.dependency_parser = dependency_parser
		self.constraints_maps = constraints_maps

	def extract(self, word_idx: int, words: ParsedText) -> Optional[Extractions]:
		return self._extract_by_constraints_maps(word_idx, words, self.constraints_maps)


class VerbPatternsArgsExtractor(PatternsArgsExtractor):
	def __init__(
			self, dependency_parser: DependencyParser,
			**kwargs):
		from yet_another_verb.configuration.extractors_config import VERB_PATTERNS
		super().__init__(dependency_parser, VERB_PATTERNS, **kwargs)

	def extract(self, word_idx: int, words: ParsedText) -> Optional[Extractions]:
		extractions = super().extract(word_idx, words)
		filtered_extractions = []

		# In passive form, prefer "by" for Subject instead for PP
		for ext in extractions:
			predicate_relatives = ext.words[ext.predicate_idx].children
			is_passive_form = any(rel.dep == DepRelation.AUXPASS for rel in predicate_relatives)
			by_pp_args = [arg for arg in ext.args if arg.arg_type == ArgumentType.PP and ext.words[arg.start_idx].text == "by"]

			if not is_passive_form or len(by_pp_args) == 0:
				filtered_extractions.append(ext)

		return filtered_extractions
