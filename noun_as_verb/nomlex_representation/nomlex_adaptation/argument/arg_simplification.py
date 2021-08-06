from copy import deepcopy
from typing import Union

from noun_as_verb.nomlex_representation.lexicon_constants.argument_value import ARGUMENT_VALUES
from noun_as_verb.nomlex_representation.lexicon_constants import LexiconType, LexiconTag, NomTypeProperty, EntryProperty
from noun_as_verb.nomlex_representation.lexicon_constants import SubcatType, ArgumentType, ArgumentValue
from noun_as_verb.nomlex_representation.nomlex_adaptation.argument.arg_renaming import rename_argument

P_DIR_REPLACEMENTS = [
	"down", "across", "from", "into", "off", "onto",
	"out", "out of", "outside", "over", "past", "through",
	"to", "toward", "towards", "up", "up to", "via"
]

P_LOC_REPLACEMENTS = [
	"above", "about", "against", "along", "around", "behind", "below", "beneath",
	"beside", "between", "beyond", "by", "in", "inside", "near", "next to", "on",
	"throughout", "under", "upon", "within"
]

NONE_VALUES = ["none", "*none*", "none"]


def simplify_argument(arg_info: Union[str, list, dict]) -> list:
	"""
	Simplies a given info of and arranges them as a list
	:param arg_info: a lexical information about an argument
	:return: an arranged list containing the leftout options
	"""

	organized_values = []

	if isinstance(arg_info, str):
		if arg_info.lower() in NONE_VALUES:
			organized_values = [arg_info]
		else:
			assert f"Illegal complement value: {arg_info}"

	elif isinstance(arg_info, list):
		organized_values = arg_info

	elif isinstance(arg_info, dict):
		for k, v in arg_info.items():
			if k in ["PP", "IND-OBJ-OTHER"]:
				p_val = v["PVAL"]
				organized_values += p_val if isinstance(p_val, list) else [p_val]
			elif k in ARGUMENT_VALUES + NONE_VALUES:
				organized_values.append(k)
			else:
				raise NotImplementedError()

	if LexiconTag.P_DIR in organized_values:
		organized_values.remove(LexiconTag.P_DIR)
		organized_values += P_DIR_REPLACEMENTS

	if LexiconTag.P_LOC in organized_values:
		organized_values.remove(LexiconTag.P_LOC)
		organized_values += P_LOC_REPLACEMENTS

	for i, v in enumerate(organized_values):
		# PP-OF -> of (Relevant for NOMLEX-2001)
		if v.startswith("PP-"):
			organized_values[i] = v.replace("PP-", "").lower()
		elif v.startswith(ArgumentType.IND_OBJ + "-"):
			organized_values[i] = v.replace(ArgumentType.IND_OBJ + "-", "").lower()

		if v.lower() in NONE_VALUES:
			organized_values[i] = ArgumentValue.NONE
		elif v in ARGUMENT_VALUES:
			organized_values[i] = ArgumentValue(v)

	return list(set(organized_values))


def simplify_subject(entry: dict, subcat: dict, lexicon_type: LexiconType):
	if lexicon_type == LexiconType.VERB:
		subcat[ArgumentType.SUBJ] = [ArgumentValue.NSUBJ, "by"]
		return

	if lexicon_type != LexiconType.NOUN:
		return

	# Adding the default subject list in case it wasn't over-written
	default_subject = entry.get(EntryProperty.VERB_SUBJ, {})
	if ArgumentType.SUBJ not in subcat.keys() and default_subject != {}:
		subcat[ArgumentType.SUBJ] = deepcopy(default_subject)

	# Updating the subject entry to be a list of possible positions
	if ArgumentType.SUBJ in subcat.keys():
		subcat[ArgumentType.SUBJ] = simplify_argument(subcat[ArgumentType.SUBJ])

		# Avoiding the NOT-PP-BY tag
		if LexiconTag.NOT_PP_BY in subcat[ArgumentType.SUBJ]:
			subcat[ArgumentType.SUBJ].remove(LexiconTag.NOT_PP_BY)

		# Avoiding the BY preposition default for subjects
		elif "by" not in subcat[ArgumentType.SUBJ]:
			subcat[ArgumentType.SUBJ].append("by")


def simplify_object(subcat: dict, subcat_type: SubcatType, lexicon_type: LexiconType):
	if lexicon_type == LexiconType.VERB:
		if SubcatType.is_transitive(subcat_type):
			subcat[ArgumentType.OBJ] = [ArgumentValue.DOBJ, ArgumentValue.NSUBJPASS]
		return

	if lexicon_type != LexiconType.NOUN:
		return

	if ArgumentType.OBJ in subcat.keys():
		subcat[ArgumentType.OBJ] = simplify_argument(subcat[ArgumentType.OBJ])


def simplify_ind_object(subcat: dict, subcat_type: SubcatType, lexicon_type: LexiconType):
	if "NP-NP" in subcat_type:
		arg_values = []
	elif "NP-TO-NP" in subcat_type:
		arg_values = ["to"]
	elif "NP-FOR-NP" in subcat_type:
		arg_values = ["for"]
	else:  # Other subcats shouldn't demand indirect objects
		return

	if lexicon_type == LexiconType.VERB:
		arg_values += [ArgumentValue.IOBJ]
	elif lexicon_type == LexiconType.NOUN:
		arg_values += simplify_argument(subcat.get(ArgumentType.IND_OBJ, []))
	else:
		return

	if len(arg_values) > 0:
		subcat[ArgumentType.IND_OBJ] = arg_values


def simplify_preps(subcat: dict, subcat_type: SubcatType, lexicon_type: LexiconType):
	for pval_type in [LexiconTag.PVAL, LexiconTag.PVAL1, LexiconTag.PVAL2]:
		nom_pval_type = pval_type.replace(LexiconTag.PVAL.value, LexiconTag.PVAL_NOM.value)
		nom_pval_role_type = "NOM-ROLE-FOR-" + pval_type
		new_pval_type = pval_type.replace(LexiconTag.PVAL.value, ArgumentType.PP.value)
		extra_values = []

		# add noun specific types of arguments
		if lexicon_type == LexiconType.NOUN:
			if nom_pval_type in subcat:
				pval_type = nom_pval_type

			extra_values += subcat.pop(nom_pval_role_type, [])
		else:
			subcat.pop(nom_pval_type, None)
			subcat.pop(nom_pval_role_type, None)

		rename_argument(pval_type, new_pval_type, subcat)

		values = simplify_argument(subcat.get(new_pval_type, [])) + simplify_argument(extra_values)
		if len(values) > 0:
			subcat[new_pval_type] = values

	# Fixing the lack of one preposition, when two prepositions are required
	# The values of PP2 are assumed to be the same as those of PP1 or PP
	if "PP-PP" in subcat_type:
		if ArgumentType.PP1 not in subcat:
			rename_argument(ArgumentType.PP, ArgumentType.PP1, subcat)

		if ArgumentType.PP2 not in subcat:
			subcat[ArgumentType.PP2] = subcat[ArgumentType.PP1]


def simplify_particle(entry: dict, subcat: dict, lexicon_type: LexiconType):
	if lexicon_type == LexiconType.VERB:
		if LexiconTag.ADVAL in subcat:
			rename_argument(LexiconTag.ADVAL, ArgumentType.PART, subcat)

		nom_type_part_value = entry[EntryProperty.NOM_TYPE][NomTypeProperty.PART]
		if nom_type_part_value is not None:
			subcat[ArgumentType.PART] = subcat.get(ArgumentType.PART, []) + [nom_type_part_value]
			subcat[ArgumentType.PART] = list(set(subcat[ArgumentType.PART]))

	elif lexicon_type == LexiconType.NOUN:
		subcat.pop(ArgumentType.PART, None)
		subcat.pop(LexiconTag.ADVAL, None)
