from itertools import chain, permutations, product
from typing import List, Optional

from yet_another_verb.arguments_extractor.args_extractor import ArgsExtractor
from yet_another_verb.arguments_extractor.extraction.extracted_argument import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.extraction import Extraction
from yet_another_verb.arguments_extractor.extraction.filters import choose_longest, uniqify
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.dependency_parser.input_text import InputText
from yet_another_verb.dependency_parsing.dependency_parser.parsed_word import ParsedWord
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.nomlex.nomlex_maestro import NomlexMaestro
from yet_another_verb.nomlex.nomlex_version import NomlexVersion
from yet_another_verb.nomlex.representation.constraints_map import ConstraintsMap
from yet_another_verb.configuration import NOMLEX_CONFIG, PARSING_CONFIG


class NomlexArgsExtractor(ArgsExtractor):
	def __init__(
			self,
			nomlex_version: NomlexVersion = NOMLEX_CONFIG.NOMLEX_VERSION,
			dependency_parser: DependencyParser = PARSING_CONFIG.DEFAULT_PARSER_MAKER()
	):
		self.adapted_lexicon = NomlexMaestro(nomlex_version).get_adapted_lexicon()
		self.dependency_parser = dependency_parser

	def preprocess(self, text: InputText) -> ParsedText:
		return self.dependency_parser(text)

	@staticmethod
	def _is_empty_or_contain(values: list, v):
		return len(values) == 0 or v in values

	def _is_word_match_constraints(self, word: ParsedWord, constraint_map: ConstraintsMap) -> bool:
		required_contraints = [
			lambda: self._is_empty_or_contain(constraint_map.values, word.text),
			lambda: self._is_empty_or_contain(constraint_map.postags, word.tag),
			lambda: self._is_empty_or_contain(constraint_map.word_relations, word.dep)
		]

		return all(constraint() is True for constraint in required_contraints)

	def _get_relatives_matched_args(
			self, relatives: List[Optional[ParsedWord]], constraints_maps: List[ConstraintsMap], cached_results: dict
	) -> Optional[List[List[ExtractedArgument]]]:
		relatives_matched_args = []
		for i, relative in enumerate(relatives):
			if i >= len(constraints_maps):
				break

			constraints_map = constraints_maps[i]
			if relative is None:
				if constraints_map.required:
					return None

				continue

			relative_matched_args = self._get_matched_arguments(relative, constraints_map, cached_results)
			if relative_matched_args is None:
				return None

			relatives_matched_args.append(relative_matched_args)

		return [list(chain(*combined_args)) for combined_args in product(*relatives_matched_args)]

	def _get_matched_arguments(
			self, word: ParsedWord, constraints_map: ConstraintsMap, cached_results: dict
	) -> Optional[List[List[ExtractedArgument]]]:
		if (word, constraints_map) in cached_results:
			return cached_results[(word, constraints_map)]

		is_match_constraints = self._is_word_match_constraints(word, constraints_map)
		if not is_match_constraints and constraints_map.required:
			return None

		relatives = list(word.children)
		relatives_constraints = constraints_map.relatives_constraints
		required_maps = [m for m in relatives_constraints if m.required]
		if len(relatives) < len(required_maps):
			return None

		arg = None
		if is_match_constraints and constraints_map.arg_type is not None:
			arg_idxs = word.subtree_indices if constraints_map.word_relations != [] else []
			arg_idxs += [word.i]
			arg = ExtractedArgument(
				arg_idxs=arg_idxs,
				arg_type=constraints_map.arg_type
			)

		matched_args_combinations = []
		relatives += [None] * (len(relatives_constraints) - len(relatives))
		relatives_permutations = set(permutations(relatives, len(relatives_constraints)))
		for relatives_permute in relatives_permutations:
			relatvies_matched_args_combinations = self._get_relatives_matched_args(
				relatives=list(relatives_permute), constraints_maps=relatives_constraints,
				cached_results=cached_results
			)

			if relatvies_matched_args_combinations is not None:
				matched_args_combinations += relatvies_matched_args_combinations

		if is_match_constraints and constraints_map.arg_type is not None:
			for args in matched_args_combinations:
				other_args_idxs = chain(*[arg.arg_idxs for arg in args])
				arg.arg_idxs = [i for i in arg.arg_idxs if i not in other_args_idxs]
				args.append(arg)

		cached_results[(word, constraints_map)] = matched_args_combinations
		return matched_args_combinations

	def extract(self, word_idx: int, parsed_text: ParsedText) -> List[Extraction]:
		word = parsed_text[word_idx]
		word_entry = self.adapted_lexicon.entries.get(word.lemma)

		if word_entry is None:
			return []

		cached_results = {}
		extractions = []
		maps = set(chain(*word_entry.subcats.values()))
		for constraint_map in maps:
			matched_args_combinations = self._get_matched_arguments(word, constraint_map, cached_results)

			if matched_args_combinations is None:
				continue

			for matched_args in matched_args_combinations:
				extractions.append(Extraction(predicate_idx=word_idx, args=list(matched_args)))

		return choose_longest(uniqify(extractions))
