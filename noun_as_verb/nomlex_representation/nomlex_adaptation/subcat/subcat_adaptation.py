from itertools import product
from copy import deepcopy
from typing import List

from noun_as_verb.nomlex_representation.lexicon_constants import LexiconType, LexiconTag, EntryProperty, SubcatType, \
	SubcatProperty, ArgumentType, ArgumentValue, UDRelation
from noun_as_verb.nomlex_representation.lexicon_constants.postag import NOUN_POSTAGS, VERB_POSTAGS
from noun_as_verb.nomlex_representation.lexicon_constants.subcat_property import SUBCAT_PROPERTIES
from noun_as_verb.nomlex_representation.constraints_map import ConstraintsMap
from noun_as_verb.nomlex_representation.nomlex_adaptation.argument.arg_enrichment import enrich_arguments
from noun_as_verb.nomlex_representation.nomlex_adaptation.argument.arg_renaming import rename_argument
from noun_as_verb.nomlex_representation.nomlex_adaptation.argument.arg_omission import should_ommit_argument
from noun_as_verb.nomlex_representation.nomlex_adaptation.argument.arg_adaptation import adapt_argument
from noun_as_verb.nomlex_representation.nomlex_adaptation.modifications import get_arg_constraints_maps
from noun_as_verb.nomlex_representation.nomlex_adaptation.subcat.subcat_simplification import simplify_subcat


def _perform_alternation_if_needed(subcat: dict, subcat_type: SubcatType):
	if subcat.pop(SubcatProperty.ALTERNATES, "F") != "T":
		return

	# SUBJ-IND-OBJ-ALT (transitive -> ditransitive); SUBJ-OBJ-ALT (intransitive -> transitive)
	new_type = ArgumentType.IND_OBJ if SubcatType.is_transitive(subcat_type) else ArgumentType.OBJ
	old_type = ArgumentType.SUBJ

	assert old_type in subcat

	# Avoid alternation when the target argument is also required for the subcat structure
	if new_type not in subcat:
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


def _is_compatible_with_not(subcat: dict, value_by_arg: dict) -> bool:
	for not_structure in subcat[SubcatProperty.NOT]:
		# Not compatible if NOT demands arg that isn't specified in the given structure
		compatible_with_not = set(not_structure.keys()).issubset(value_by_arg.keys())

		for arg_type, arg_value in value_by_arg.items():
			if arg_type not in not_structure:
				continue

			# Not compatible if demanded arg values don't match
			if arg_value not in not_structure[arg_type]:
				compatible_with_not = False
			else:
				# Not compatible if demanded arg preps don't match ENTIRELY
				preps = set(subcat[SubcatProperty.ARGUMENTS][arg_type][arg_value])
				not_preps = not_structure[arg_type][arg_value]
				without_not_preps = preps - set(not_preps)
				if (len(preps) > 0 or len(not_preps) > 0) and len(without_not_preps) > 0:
					subcat[SubcatProperty.ARGUMENTS][arg_type][arg_value] = without_not_preps
					compatible_with_not = False

		if compatible_with_not:
			return True

	return False


def _should_ommit_arg_combination(subcat: dict, value_by_arg: dict) -> bool:
	if _is_compatible_with_not(subcat, value_by_arg):
		return True

	values = list(value_by_arg.values())
	repeated_values = set([x for x in values if values.count(x) > 1])

	# Only NP value can specified to multiple arguments
	return len(repeated_values) > 0 and not repeated_values.issubset({ArgumentValue.NP})


def _get_args_combinations(subcat: dict):
	arg_types = list(subcat[SubcatProperty.ARGUMENTS].keys())
	args_values = [subcat[SubcatProperty.ARGUMENTS][arg_type] for arg_type in arg_types]
	all_combinations = [dict(zip(arg_types, v)) for v in product(*args_values)]
	relavant_combinations = []

	for value_by_arg in all_combinations:
		if not _should_ommit_arg_combination(subcat, value_by_arg):
			relavant_combinations.append(value_by_arg)

	return relavant_combinations


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
	for value_by_arg in _get_args_combinations(subcat):
		passive_voice = \
			lexicon_type == LexiconType.VERB and (
				value_by_arg.get(ArgumentType.SUBJ, None) == ArgumentValue.NP or
				value_by_arg.get(ArgumentType.OBJ, None) == ArgumentValue.NSUBJPASS
			)

		arg_constraints = []
		for arg_type, arg_value in value_by_arg.items():
			if _is_value_required_by_other_arg(subcat, arg_type, arg_value):
				continue

			# TODO: handle specific ING subcats differently
			arg_constraints += get_arg_constraints_maps(
				subcat_type=subcat_type, arg_type=arg_type, arg_value=arg_value, lexicon_type=lexicon_type,
				preps=subcat[SubcatProperty.ARGUMENTS][arg_type][arg_value],
				is_required=arg_type in subcat[SubcatProperty.REQUIRED]
			)

		extra_constraints = [ConstraintsMap(ud_relations=[UDRelation.AUXPASS])] if passive_voice else []

		predicate_values = []
		if lexicon_type == LexiconType.NOUN:
			predicate_values += [entry[EntryProperty.ORTH]] if not is_plural_only else []
			predicate_values += [entry[EntryProperty.PLURAL]] if not is_singular_only else []

		subcat_maps.append(ConstraintsMap(
			ud_relations=[UDRelation.SELF_RELATION],
			postags=VERB_POSTAGS if lexicon_type == LexiconType.VERB else NOUN_POSTAGS,
			values=predicate_values,
			sub_constraints=arg_constraints + extra_constraints)
		)

	return subcat_maps


def reconstruct_constraints(entry: dict, subcat: dict, subcat_type: SubcatType, lexicon_type: LexiconType):
	_perform_alternation_if_needed(subcat, subcat_type)
	_combine_args_into_property(subcat, subcat_type, lexicon_type)
	entry[EntryProperty.SUBCATS][subcat_type] = _get_subcat_as_constraint_maps(entry, subcat, subcat_type, lexicon_type)


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
