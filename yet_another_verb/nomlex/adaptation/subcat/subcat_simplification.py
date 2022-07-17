from typing import List

from yet_another_verb.nomlex.constants import LexiconType, LexiconTag, \
	EntryProperty, SubcatType, SubcatProperty, ArgumentValue
from yet_another_verb.nomlex.adaptation.argument.arg_renaming import rename_argument
from yet_another_verb.nomlex.adaptation.argument.arg_simplification import simplify_subject, \
	simplify_object, simplify_preps, simplify_ind_object, simplify_particle, simplify_argument
from yet_another_verb.nomlex.adaptation.argument.arg_adaptation import get_adapted_argument
from yet_another_verb.nomlex.adaptation.modifications import get_default_requires, \
	get_default_optionals, get_argument_renamings, get_extra_not_constraints
from yet_another_verb.arguments_extractor.extraction import ArgumentType


def _is_subcat_requires_object(entry: dict, subcat_type: SubcatType):
	if not SubcatType.is_transitive(subcat_type):
		return False

	if ArgumentType.OBJ in entry[EntryProperty.SUBCATS][subcat_type][SubcatProperty.REQUIRED]:
		return True

	other_subcat_types = set(entry.keys())

	# OBJECT is optional for NOM-NP-X subcats, only if NOM-X isn't compatible with the current entry
	if subcat_type == SubcatType.NOM_NP:
		return not other_subcat_types.isdisjoint({
			SubcatType.NOM_INTRANS, SubcatType.NOM_INTRANS_RECIP
		})

	if subcat_type == SubcatType.NOM_NP_AS_NP_SC:
		return SubcatType.NOM_AS_NP in other_subcat_types

	subcat_without_np = subcat_type.replace("NOM-PART-NP", "NOM-PART")
	subcat_without_np = subcat_without_np.replace("NOM-NP-", "NOM-")
	return subcat_without_np in other_subcat_types


def rename_arguments(subcat: dict, subcat_type: SubcatType, lexicon_type: LexiconType):
	names_mapping = get_argument_renamings(subcat_type, lexicon_type)

	for old_type, new_type in names_mapping.items():
		rename_argument(old_type, new_type, subcat)


def simplify_arguments(entry: dict, subcat: dict, subcat_type: SubcatType, lexicon_type: LexiconType):
	simplify_subject(entry, subcat, lexicon_type)
	simplify_object(subcat, subcat_type, lexicon_type)
	simplify_ind_object(subcat, subcat_type, lexicon_type)
	simplify_preps(subcat, subcat_type, lexicon_type)
	simplify_particle(entry, subcat, lexicon_type)
	rename_arguments(subcat, subcat_type, lexicon_type)


def add_args_requirement(arg_types: List[ArgumentType], subcat: dict, force: bool = False, as_optional: bool = False):
	old_property = SubcatProperty.REQUIRED if as_optional else SubcatProperty.OPTIONAL
	new_property = SubcatProperty.OPTIONAL if as_optional else SubcatProperty.REQUIRED

	for arg_type in arg_types:
		if force or arg_type not in subcat[old_property]:
			subcat[new_property] = list(set(subcat[new_property] + [arg_type]))
			subcat[old_property] = list(set(subcat[old_property]) - {arg_type})


def simplify_requires_and_optionals(entry: dict, subcat: dict, subcat_type: SubcatType, lexicon_type: LexiconType):
	original_requirements = subcat.pop(SubcatProperty.REQUIRED, {})
	original_optionals = subcat.pop(SubcatProperty.OPTIONAL, {})

	# The construction of optionals and requires is done at a certain order
	# according to the reliability of the data in the subcat info

	# Arguments with None value are optional + original optional arguments (#1)
	subcat[SubcatProperty.OPTIONAL] = list(original_optionals.keys()) + \
		[c for c, v in subcat.items() if isinstance(v, list) and ArgumentValue.NONE in v]
	subcat[SubcatProperty.REQUIRED] = []

	# Extracting requires info from REQUIRED (#2)
	for complement_type, require_info in original_requirements.items():
		if len(require_info.keys()) == 0:
			add_args_requirement([complement_type], subcat)

		if LexiconTag.DET_POSS_ONLY in require_info:
			subcat[SubcatProperty.DET_POSS_NO_OTHER_OBJ].append(complement_type)

		if LexiconTag.N_N_MOD_ONLY in require_info:
			subcat[SubcatProperty.N_N_MOD_NO_OTHER_OBJ].append(complement_type)

	# Use default requirements for this subcat type (#3)
	add_args_requirement(get_default_requires(subcat_type, lexicon_type), subcat)
	add_args_requirement(get_default_optionals(subcat_type, lexicon_type), subcat, as_optional=True)

	# Check if the subcat type requires an object (#4)
	if _is_subcat_requires_object(entry, subcat_type):
		add_args_requirement([ArgumentType.OBJ], subcat)
	elif SubcatType.is_transitive(subcat_type) and ArgumentType.OBJ in subcat:
		add_args_requirement([ArgumentType.OBJ], subcat, as_optional=True)

	# SUBJECT is optional by default (#5)
	add_args_requirement([ArgumentType.SUBJ], subcat, as_optional=True)

	# particle is required if available
	if ArgumentType.PART in subcat:
		add_args_requirement([ArgumentType.PART], subcat, force=True)

	assert set(subcat[SubcatProperty.REQUIRED]).isdisjoint(subcat[SubcatProperty.OPTIONAL])


def simplify_not(subcat: dict, subcat_type: SubcatType, lexicon_type: LexiconType):
	new_not_info = []
	for values_per_arg in subcat.pop(SubcatProperty.NOT, {}).values():
		simplify_preps(values_per_arg, subcat_type, lexicon_type)
		constraints_per_arg = {}

		for arg_type, arg_info in values_per_arg.items():
			arg_info = simplify_argument(arg_info)

			# Ignore constraints of arguments that aren't relevant for verbs
			if lexicon_type != LexiconType.VERB or arg_info not in [ArgumentType.SUBJ, ArgumentType.OBJ]:
				adapted_arg = get_adapted_argument(subcat_type, arg_type, arg_info, lexicon_type)

				if len(adapted_arg) > 0:
					constraints_per_arg[arg_type] = adapted_arg

		# Add constraints that consider at least two arguments
		if len(constraints_per_arg.keys()) > 1:
			new_not_info.append(constraints_per_arg)

	subcat[SubcatProperty.NOT] = new_not_info + get_extra_not_constraints(subcat_type, lexicon_type)


def simplify_subcat(entry: dict, subcat: dict, subcat_type: SubcatType, lexicon_type: LexiconType):
	simplify_arguments(entry, subcat, subcat_type, lexicon_type)
	simplify_requires_and_optionals(entry, subcat, subcat_type, lexicon_type)
	simplify_not(subcat, subcat_type, lexicon_type)
