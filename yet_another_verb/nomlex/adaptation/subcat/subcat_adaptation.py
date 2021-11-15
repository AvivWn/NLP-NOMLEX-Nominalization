from collections import namedtuple
from itertools import product
from copy import deepcopy
from typing import List, Dict, Callable

from yet_another_verb.nomlex.constants import LexiconType, LexiconTag, EntryProperty, SubcatType, \
	SubcatProperty, ArgumentType, ArgumentValue, WordRelation
from yet_another_verb.nomlex.constants.word_postag import NOUN_POSTAGS, VERB_POSTAGS
from yet_another_verb.nomlex.constants.subcat_property import SUBCAT_PROPERTIES
from yet_another_verb.nomlex.representation.constraints_map import ConstraintsMap
from yet_another_verb.nomlex.adaptation.argument.arg_enrichment import enrich_arguments
from yet_another_verb.nomlex.adaptation.argument.arg_renaming import rename_argument
from yet_another_verb.nomlex.adaptation.argument.arg_omission import should_ommit_argument
from yet_another_verb.nomlex.adaptation.argument.arg_adaptation import adapt_argument
from yet_another_verb.nomlex.adaptation.modifications import get_arg_constraints_maps, get_arg_attributes_property, \
	get_maps_with_complex_constraints
from yet_another_verb.nomlex.adaptation.subcat.subcat_simplification import simplify_subcat
from yet_another_verb.nomlex.representation.lexical_argument import LexicalArgument
from yet_another_verb.nomlex.representation.lexical_subcat import LexicalSubcat
from yet_another_verb.nomlex.adaptation.modifications.arg_properties import get_plural_property, \
	get_subjunct_property, get_controlled_args

ArgValuePair = namedtuple('ArgValuePair', ['value', 'preps'])
ArgCombination = Dict[ArgumentType, ArgValuePair]


def combine_keys_multi_values(keys: list, multi_values: List[list], value_func: Callable):
	assert len(multi_values) == len(keys)
	return [dict(zip(keys, list(map(lambda v: value_func(v), values)))) for values in product(*multi_values)]


def _perform_alternation_if_needed(subcat: dict, subcat_type: SubcatType):
	if subcat.pop(SubcatProperty.ALTERNATES, "F") != "T":
		return

	# SUBJ-IND-OBJ-ALT (transitive -> ditransitive); SUBJ-OBJ-ALT (intransitive -> transitive)
	new_type = ArgumentType.IND_OBJ if SubcatType.is_transitive(subcat_type) else ArgumentType.OBJ
	old_type = ArgumentType.SUBJ

	assert old_type in subcat

	# Avoid alternation when the target argument is also required for the subcat structure
	if new_type in subcat:
		return

	# Avoid alternation when argument corresponds to the nominalization (like renter or boiler)
	# In such cases, ALTERNATES only affect the suitable verb
	if ArgumentValue.NOM in subcat[old_type]:
		return

	rename_argument(old_type, new_type, subcat)


def _combine_args_into_property(subcat: dict, subcat_type: SubcatType, lexicon_type: LexiconType):
	arguments = {}
	for i in deepcopy(subcat):
		if isinstance(i, SubcatProperty):
			continue

		assert i not in SUBCAT_PROPERTIES

		# Otherwise the property should be an argument
		if should_ommit_argument(i):
			subcat.pop(i)
			continue

		adapt_argument(subcat, subcat_type, ArgumentType(i), lexicon_type)

		if i in subcat:
			arguments[ArgumentType(i)] = subcat.pop(i)

	# Aggregate all arguments under one property
	subcat[SubcatProperty.ARGUMENTS] = arguments
	assert set(subcat[SubcatProperty.REQUIRED] + subcat[SubcatProperty.OPTIONAL]) == set(subcat[SubcatProperty.ARGUMENTS].keys())


def _is_compatible_with_not(subcat: dict, pair_by_arg: dict) -> bool:
	for not_structure in subcat[SubcatProperty.NOT]:
		# Not compatible if NOT demands arg that isn't specified in the given structure
		compatible_with_not = set(not_structure.keys()).issubset(pair_by_arg.keys())

		for arg_type, arg_pair in pair_by_arg.items():
			arg_value, arg_preps = arg_pair

			if arg_type not in not_structure:
				continue

			# Not compatible if demanded arg values don't match
			if arg_value not in not_structure[arg_type]:
				compatible_with_not = False
			else:
				# Not compatible if demanded arg preps don't match ENTIRELY
				not_preps = not_structure[arg_type][arg_value]
				without_not_preps = set(arg_preps) - set(not_preps)
				if (len(arg_preps) > 0 or len(not_preps) > 0) and len(without_not_preps) > 0:
					# pair_by_arg[arg_type] = (arg_value, without_not_preps)
					compatible_with_not = False

		if compatible_with_not:
			return True

	return False


def _split_arg_combination_by_preps(combination: ArgCombination):
	splitted_combination = {}
	for arg in combination:
		arg_value_pair = combination[arg].value

		value_pairs = [ArgValuePair(arg_value_pair, [prep]) for prep in combination[arg][1]]
		if len(combination[arg].preps) == 0:
			value_pairs = [ArgValuePair(arg_value_pair, [])]

		splitted_combination[arg] = value_pairs

	return combine_keys_multi_values(list(combination.keys()), list(splitted_combination.values()), lambda x: x)


def _filter_args_combinations(subcat: dict, args_combinations: List[ArgCombination]) -> List[dict]:
	to_check_combinations = []

	for combination in args_combinations:
		# Separate combination by prepositions, if NOT constraint exists
		if len(subcat[SubcatProperty.NOT]) > 0:
			to_check_combinations += _split_arg_combination_by_preps(combination)
			# to_check_combinations += [dict(zip(combination, t)) for t in product(*combination.values())]
		else:
			to_check_combinations.append(combination)

	filtered_combinations = []
	for combination in to_check_combinations:
		# Check for NOT constraint
		if _is_compatible_with_not(subcat, combination):
			continue

		# Only NP value can specified to multiple arguments
		values = list(combination.values())
		repeated_values = [x for x in values if values.count(x) > 1 and x.value != ArgumentValue.NP]
		if len(repeated_values) > 0:
			continue

		filtered_combinations.append(combination)

	return filtered_combinations


def _get_args_combinations(subcat: dict) -> list:
	arg_types = list(subcat[SubcatProperty.ARGUMENTS].keys())
	args_value_pairs = [subcat[SubcatProperty.ARGUMENTS][arg_type].items() for arg_type in arg_types]
	all_combinations = combine_keys_multi_values(arg_types, args_value_pairs, lambda x: ArgValuePair(*x))
	return _filter_args_combinations(subcat, all_combinations)


def _is_value_required_by_other_arg(subcat: dict, arg_type: ArgumentType, arg_value: ArgumentValue) -> bool:
	required_args_by_value = {
		ArgumentValue.N_N_MOD: subcat[SubcatProperty.N_N_MOD_NO_OTHER_OBJ],
		ArgumentValue.DET_POSS: subcat[SubcatProperty.DET_POSS_NO_OTHER_OBJ]
	}

	if arg_value not in required_args_by_value:
		return False

	required_args = required_args_by_value[arg_value]
	return arg_type not in required_args and len(required_args) > 0


def _get_subcat_as_constraint_maps(
		entry: dict, subcat: dict, subcat_type: SubcatType, lexicon_type: LexiconType
) -> List[ConstraintsMap]:
	is_singular_only = LexiconTag.SING_ONLY in entry[EntryProperty.NOUN] or entry.get(EntryProperty.PLURAL) is None
	is_plural_only = entry.get(EntryProperty.SINGULAR_FALSE, False) or LexiconTag.PLUR_ONLY in entry[EntryProperty.NOUN]
	assert not (is_singular_only and is_plural_only)

	subcat_maps = []
	for value_pair_by_arg in _get_args_combinations(subcat):
		value_by_arg, preps_by_arg = {}, {}

		passive_voice = \
			lexicon_type == LexiconType.VERB and (
				value_pair_by_arg.get(ArgumentType.SUBJ, None) == ArgumentValue.NP or
				value_pair_by_arg.get(ArgumentType.OBJ, None) == ArgumentValue.NSUBJPASS
			)

		nom_arg_type = None
		constraints_by_arg = {}
		for arg_type, arg_value_pair in value_pair_by_arg.items():
			value_by_arg[arg_type], preps_by_arg[arg_type] = arg_value_pair.value, arg_value_pair.preps

			if _is_value_required_by_other_arg(subcat, arg_type, arg_value_pair.value):
				continue

			if arg_value_pair.value is ArgumentValue.NOM:
				nom_arg_type = arg_type
				continue

			constraints_by_arg[arg_type] = get_arg_constraints_maps(
				arg_type=arg_type, arg_value=arg_value_pair.value, lexicon_type=lexicon_type,
				preps=arg_value_pair.preps,
				is_required=arg_type in subcat[SubcatProperty.REQUIRED],
			)
			assert constraints_by_arg[arg_type] != []

		extra_constraints = [ConstraintsMap(word_relations=[WordRelation.AUXPASS])] if passive_voice else []
		predicate_values = []
		if lexicon_type == LexiconType.NOUN:
			predicate_values += [entry[EntryProperty.ORTH]] if not is_plural_only else []
			predicate_values += [entry[EntryProperty.PLURAL]] if not is_singular_only else []

		predicate_map = ConstraintsMap(
			arg_type=nom_arg_type,
			postags=VERB_POSTAGS if lexicon_type == LexiconType.VERB else NOUN_POSTAGS,
			values=predicate_values,
			relatives_constraints=extra_constraints
		)

		for combined_constraints in product(*constraints_by_arg.values()):
			map_by_arg = dict(zip(constraints_by_arg.keys(), combined_constraints))
			combined_constraints_options = [list(combined_constraints)] + \
				get_maps_with_complex_constraints(subcat_type, map_by_arg, value_by_arg, preps_by_arg, lexicon_type)

			for constraints in combined_constraints_options:
				expanded_map = deepcopy(predicate_map)
				expanded_map.relatives_constraints += constraints
				subcat_maps.append(expanded_map)

	return list(set(subcat_maps))


def reconstruct_constraints(entry: dict, subcat: dict, subcat_type: SubcatType, lexicon_type: LexiconType):
	_perform_alternation_if_needed(subcat, subcat_type)
	_combine_args_into_property(subcat, subcat_type, lexicon_type)

	lexical_args = {}
	for arg_type in subcat[SubcatProperty.ARGUMENTS]:
		lexical_args[arg_type] = LexicalArgument(
			arg_type=arg_type,
			plural=get_plural_property(subcat_type, arg_type),
			subjunct=get_subjunct_property(subcat_type, arg_type),
			controlled=get_controlled_args(subcat_type, arg_type),
			attributes=entry.get(get_arg_attributes_property(arg_type), [])
		)

	entry[EntryProperty.SUBCATS][subcat_type] = LexicalSubcat(
		subcat_type=subcat_type,
		constraints_maps=_get_subcat_as_constraint_maps(entry, subcat, subcat_type, lexicon_type),
		lexical_args=lexical_args
	)


def adapt_subcat(entry: dict, subcat_type: SubcatType, lexicon_type: LexiconType):
	"""
	Adapt and simplify the constraints of the given subcategorization, as specified in the given entry
	:param entry: a nomlex entry as a json format
	:param subcat_type: the type of the subcategorization
	:param lexicon_type: the type of the lexicon
	"""

	subcat = entry[EntryProperty.SUBCATS][subcat_type]
	simplify_subcat(entry, subcat, subcat_type, lexicon_type)
	enrich_arguments(entry, subcat, subcat_type, lexicon_type)
	reconstruct_constraints(entry, subcat, subcat_type, lexicon_type)
