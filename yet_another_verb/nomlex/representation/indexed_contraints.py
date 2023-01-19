from collections import defaultdict, namedtuple
from itertools import product
from typing import List, TYPE_CHECKING, Iterator, Optional, Set, Dict

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from yet_another_verb.dependency_parsing.dependency_parser.parsed_word import ParsedWord
from yet_another_verb.utils.print_utils import print_if_verbose

if TYPE_CHECKING:
	from yet_another_verb.nomlex.representation.constraints_map import ConstraintsMap

WordProperty = namedtuple("WordProperty", ["value", "postag", "relation"])


@dataclass_json
@dataclass
class IndexedConstraintsMaps:
	constraints_maps: List['ConstraintsMap'] = field(default_factory=list)

	@staticmethod
	def _expand_with_suitable_properties(
			property_mapping: Dict[WordProperty, Set[int]],
			constraints_map: 'ConstraintsMap',
			map_idx: int):
		values = constraints_map.values if len(constraints_map.values) > 0 else [None]
		postags = constraints_map.postags if len(constraints_map.postags) > 0 else [None]
		relations = constraints_map.dep_relations if len(constraints_map.dep_relations) > 0 else [None]

		for value, postag, relation in product(values, postags, relations):
			word_property = WordProperty(value=value, postag=postag, relation=relation)
			property_mapping[word_property].add(map_idx)

	def __post_init__(self):
		self._ordered_indices = list(range(len(self.constraints_maps)))
		self._ordered_indices.sort(key=lambda c_id: len(self.constraints_maps[c_id].included_args), reverse=True)

		self._required_indices = set()
		self._all_optional_relative_indices = set()
		self._indices_by_properties = defaultdict(set)
		self._indices_by_relative_properties = defaultdict(set)

		for i, constraints_map in enumerate(self.constraints_maps):
			if constraints_map.required:
				self._required_indices.add(i)

			required_relative_maps = constraints_map.get_required_relative_maps()
			if len(required_relative_maps) == 0:
				self._all_optional_relative_indices.add(i)

			self._expand_with_suitable_properties(self._indices_by_properties, constraints_map, i)

			for relative_map in constraints_map.relatives_constraints:
				self._expand_with_suitable_properties(self._indices_by_relative_properties, relative_map, i)

	@staticmethod
	def _get_relevant_indices_in_mapping(property_mapping: Dict[WordProperty, Set[int]], word: ParsedWord) -> Set[int]:
		relevant_indices = set()
		for value, postag, relation in product([word.text, None], [word.tag, word.pos, None], [word.dep, None]):
			word_property = WordProperty(value=value, postag=postag, relation=relation)
			relevant_indices.update(property_mapping[word_property])

		return relevant_indices

	def _reorder_by_potential(self, indices: Set[int]) -> List[int]:
		ordered_indices = []
		for idx in self._ordered_indices:
			if idx in indices:
				ordered_indices.append(idx)

		return ordered_indices

	def _get_relevant_constraints_maps_indices(
			self, word: ParsedWord, relatives: Optional[List[ParsedWord]] = None,
			order_by_potential: bool = False
	) -> List[int]:
		relevant_indices = self._get_relevant_indices_in_mapping(self._indices_by_properties, word)

		if relatives is not None:
			relatives_relevant_indices = set()
			for relative in relatives:
				relatives_relevant_indices.update(
					self._get_relevant_indices_in_mapping(self._indices_by_relative_properties, relative))

			relatives_relevant_indices.update(self._all_optional_relative_indices)
			relevant_indices = relevant_indices.intersection(relatives_relevant_indices)

		if order_by_potential:
			relevant_indices = self._reorder_by_potential(relevant_indices)
		else:
			relevant_indices = list(relevant_indices)

		print_if_verbose("SAVED:", len(self.constraints_maps) - len(relevant_indices))
		return relevant_indices

	def get_relevant_constraints_maps(
			self, word: ParsedWord, order_by_potential: bool = False) -> Iterator['ConstraintsMap']:
		relevant_indices = self._get_relevant_constraints_maps_indices(word, word.children, order_by_potential)
		return map(lambda i: self.constraints_maps[i], relevant_indices)

	def get_relevant_maps_combinations(self, words: List[ParsedWord]) -> Iterator[List[Optional['ConstraintsMap']]]:
		coresponding_relevant_maps_indices = []
		for i, word in enumerate(words):
			relevant_maps_indices = self._get_relevant_constraints_maps_indices(word)
			coresponding_relevant_maps_indices.append(relevant_maps_indices + [-i])

		maps_indices_permutations = product(*coresponding_relevant_maps_indices)

		for indices_permute in maps_indices_permutations:
			if len(set(indices_permute)) < len(indices_permute):
				continue

			if not self._required_indices.issubset(indices_permute):
				continue

			yield [self.constraints_maps[i] if i >= 0 else None for i in indices_permute]
