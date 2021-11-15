from itertools import chain, permutations, product
from typing import List, Optional
from copy import deepcopy

from yet_another_verb.arguments_extractor.args_extractor import ArgsExtractor
from yet_another_verb.arguments_extractor.extraction.extracted_argument import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.extraction import Extraction, Extractions
from yet_another_verb.arguments_extractor.extraction.filters import prefer_by_n_args, uniqify, \
	prefer_by_constraints
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.dependency_parser.input_text import InputText
from yet_another_verb.dependency_parsing.dependency_parser.parsed_word import ParsedWord
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.nomlex.constants import ArgumentType
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

	@staticmethod
	def _is_not_empty_and_disjoint(s1: set, s2: set):
		return s1 != [] and set(s1).isdisjoint(s2)

	def _is_word_match_constraints(self, word: ParsedWord, constraints_map: ConstraintsMap) -> bool:
		required_contraints = [
			lambda: self._is_empty_or_contain(constraints_map.values, word.text),
			lambda: self._is_empty_or_contain(constraints_map.postags, word.tag),
			lambda: self._is_empty_or_contain(constraints_map.word_relations, word.dep)
		]

		return all(constraint() is True for constraint in required_contraints)

	@staticmethod
	def _filter_word_relatives(word: ParsedWord, constraints_map: ConstraintsMap) -> List[ParsedWord]:
		relevant_relations, relevant_postags, relevant_values = set(), set(), set()
		for m in constraints_map.relatives_constraints:
			relevant_relations.update(m.word_relations)
			relevant_postags.update(m.postags)
			relevant_values.update(m.values)

			# is JOKER
			if len(m.postags) == 0 and len(m.postags) == 0 and len(m.values) == 0:
				return [relative for relative in word.children]

		return [
			relative for relative in word.children
			if relative.dep in relevant_relations or
			   relative.tag in relevant_postags or
			   relative.text in relevant_values
		]

	def _get_relatives_matched_args(
			self, predicate: ParsedWord, relatives: List[Optional[ParsedWord]],
			constraints_maps: List[ConstraintsMap], cache: dict
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

			relative_matched_args = self._get_matched_arguments(predicate, relative, constraints_map, cache)
			if relative_matched_args is None:
				return None

			relatives_matched_args.append(relative_matched_args)

		return [list(chain(*combined_args)) for combined_args in product(*relatives_matched_args)]

	@staticmethod
	def _get_combined_args(args: List[ExtractedArgument], new_arg: ExtractedArgument, constraints_map: ConstraintsMap) -> List[ExtractedArgument]:
		combined_args = []
		other_args_idxs = set()
		for other_arg in args:
			if other_arg.arg_type == constraints_map.arg_type:
				# if new_arg.arg_type is not None:
				new_arg.arg_idxs.update(other_arg.arg_idxs)
				new_arg.fulfilled_constraints += other_arg.fulfilled_constraints
			else:
				other_args_idxs.update(other_arg.arg_idxs)
				combined_args.append(other_arg)

		new_arg.arg_idxs = new_arg.arg_idxs - other_args_idxs
		combined_args.append(new_arg)
		return combined_args

	def _has_matching_potential(self, relatives: List[ParsedWord], relatives_constraints: List[ConstraintsMap]) -> bool:
		required_constraints = [m for m in relatives_constraints if m.required]
		if len(relatives) < len(required_constraints):
			return False

		existing_relations, existing_postags, existing_values = set(), set(), set()
		for word in relatives:
			existing_relations.add(word.dep)
			existing_postags.add(word.tag)
			existing_values.add(word.text)

		required_contraints = [
			lambda x: self._is_not_empty_and_disjoint(x.values, existing_values),
			lambda x: self._is_not_empty_and_disjoint(x.word_relations, existing_relations),
			lambda x: self._is_not_empty_and_disjoint(x.postags, existing_postags)
		]

		if any(any(c(x) for c in required_contraints) for x in required_constraints):
			return False

		return True

	def _get_matched_arguments(
			self, predicate: ParsedWord, word: ParsedWord, constraints_map: ConstraintsMap, cache: dict
	) -> Optional[List[List[ExtractedArgument]]]:
		if (word, constraints_map) in cache:
			return cache[(word, constraints_map)]

		is_match_constraints = self._is_word_match_constraints(word, constraints_map)
		if not is_match_constraints and constraints_map.required:
			return None

		relatives_constraints = constraints_map.relatives_constraints
		relevant_relatives = self._filter_word_relatives(word, constraints_map)
		if not self._has_matching_potential(relevant_relatives, relatives_constraints):
			return None

		arg_idxs = set(word.subtree_indices) if predicate != word else set()
		arg = ExtractedArgument(
			arg_idxs=arg_idxs.union({word.i}),
			arg_type=constraints_map.arg_type if is_match_constraints else None,
			fulfilled_constraints=[constraints_map] if is_match_constraints else []
		)

		matched_args_combinations = []
		relevant_relatives += [None] * (len(relatives_constraints) - len(relevant_relatives))
		relatives_permutations = set(permutations(relevant_relatives, len(relatives_constraints)))
		for relatives_permute in relatives_permutations:
			relatives_matched_args_combinations = self._get_relatives_matched_args(
				predicate=predicate, relatives=list(relatives_permute),
				constraints_maps=relatives_constraints, cache=cache
			)

			if relatives_matched_args_combinations is not None:
				matched_args_combinations += relatives_matched_args_combinations

		for i, matched_args in enumerate(matched_args_combinations):
			# matched_args_combinations[i] = self._get_combined_args(matched_args, arg, constraints_map)
			matched_args_combinations[i] = self._get_combined_args(matched_args, deepcopy(arg), constraints_map)

		cache[(word, constraints_map)] = matched_args_combinations
		return matched_args_combinations

	@staticmethod
	def _filter_typless_args(extracted_args: List[ExtractedArgument]):
		return [arg for arg in extracted_args if arg.arg_type is not None]

	@staticmethod
	def _reorder_numbered_args(extracted_args: List[ExtractedArgument]):
		arg_by_type = {arg.arg_type: arg for arg in extracted_args}
		pp1_arg = arg_by_type.get(ArgumentType.PP1)
		pp2_arg = arg_by_type.get(ArgumentType.PP2)

		if pp1_arg is not None and pp2_arg is not None:
			pp1_start = min(pp1_arg.arg_idxs)
			pp2_start = min(pp2_arg.arg_idxs)

			if pp1_start > pp2_start:
				pp1_arg.arg_type = ArgumentType.PP2
				pp2_arg.arg_type = ArgumentType.PP1

	def extract(self, word_idx: int, parsed_text: ParsedText) -> Optional[Extractions]:
		word = parsed_text[word_idx]
		word_entries = self.adapted_lexicon.entries.get(word.lemma)

		if word_entries is None:
			return None

		cache = {}
		extractions = []
		max_num_of_args = 0
		for word_entry in word_entries:
			for i, constraints_map in enumerate(word_entry.constraints_maps):
				if len(constraints_map.included_args) < max_num_of_args:
					break

				matched_args_combinations = self._get_matched_arguments(word, word, constraints_map, cache)
				if matched_args_combinations is None:
					continue

				for matched_args in matched_args_combinations:
					fulfilled_constraints = list(chain(*[a.fulfilled_constraints for a in matched_args]))
					tmp_matched_args = self._filter_typless_args(matched_args)
					self._reorder_numbered_args(matched_args)

					if len(tmp_matched_args) > 0:
						extraction = Extraction(predicate_idx=word_idx, args=set(tmp_matched_args), fulfilled_constraints=fulfilled_constraints)
						extractions.append(extraction)
						max_num_of_args = max(max_num_of_args, len(tmp_matched_args))

		if len(extractions) == 0:
			return None

		for filter_func in [uniqify, prefer_by_n_args, prefer_by_constraints]:
			extractions = filter_func(extractions)

		return extractions
