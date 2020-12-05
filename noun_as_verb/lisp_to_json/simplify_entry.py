from copy import deepcopy

from .simplify_subcat import simplify_subcat
from .simplify_representation import simplify_representation
from .lexicon_modifications import lexicon_fixes_dict, subcat_typos_dict
from .utils import get_current_specs, curr_specs, is_known, unknown_values_dict, known_values_dict, get_verb_type, without_part
from noun_as_verb.utils import engine
from noun_as_verb.constants.lexicon_constants import *

# For debug
alt_subcats = {FEATURE_SUBJ_OBJ_ALT: [], FEATURE_SUBJ_IND_OBJ_ALT: []}



def split_nom_and_verb_subcat(entry, subcat, subcat_type):
	"""
	Splits the given subcat entry into two different subcats, one for verb and one for nominalization
	:param entry: an entry in the lexicon as a dictionary of dictionaries
	:param subcat: a dictionary of the subcategorization info ({ARG1: {POS1: {...}, POS2: {...}, ...}, ARG2: {...}, NOT: {...}, REQUIRED: {...}, OPTIONALS: {...}})
	:param subcat_type: the type of the subcategorization
	:return: verb's subcat and nominalization's subcat
	"""

	# Handling the nominalization subcat entry
	verb_subcat = deepcopy(subcat)
	curr_specs["is_verb"] = True
	simplify_subcat(entry, verb_subcat, subcat_type, is_verb=True)
	simplify_representation(verb_subcat, subcat_type, is_verb=True)

	# Handling the nominalization subcat entry
	nom_subcat = deepcopy(subcat)
	curr_specs["is_verb"] = False
	simplify_subcat(entry, nom_subcat, subcat_type, is_verb=False)
	simplify_representation(nom_subcat, subcat_type, is_verb=False)

	curr_specs["is_verb"] = None

	return verb_subcat, nom_subcat

def check_subcat_type(entry, subcat_type):
	"""
	Checks whether the given subcat has a typo and if it considered a known subcat
	Only subcats that appear in the fixes dictionary are consider as known (since they are the only one that we know how to handle correctly)
	The function fixes subcat types with typos and removes unknown ones.
	:param entry: an entry in the lexicon as a dictionary of dictionaries
	:param subcat_type: a string, like "NOM-NP-PP"
	:return: the subcat type after fixing typos, or None for unknown subcat types
	"""

	updated_subcat_type = subcat_type

	if without_part(subcat_type) not in lexicon_fixes_dict.keys():
		# Fixing typos in subcat names
		if subcat_type in subcat_typos_dict.keys():
			updated_subcat_type = subcat_typos_dict[subcat_type]
		else:
			# Ignoring unknown subcats
			unknown_values_dict["SUBCAT"].add(subcat_type)
			entry[ENT_VERB_SUBC].pop(subcat_type)
			return None

	# Change the wrong subcat type to the correct one
	if subcat_type != updated_subcat_type:
		entry[ENT_VERB_SUBC][updated_subcat_type] = entry[ENT_VERB_SUBC].pop(subcat_type)

	known_values_dict["SUBCAT"].add(updated_subcat_type)

	return updated_subcat_type

def rearrange_entry_properties(entry):
	"""
	Rearranges the properties of the given entry (like OBJ-ATTRIBUTE and NOM-TYPE but mainly VERB-SUBC)
	:param entry: an entry in the lexicon as a dictionary of dictionaries
	:return: None
	"""

	nom_features = entry.get(ENT_FEATURES, {}).keys()

	# Remove subentries that are not in use
	entry.pop(ENT_NOUN_SUBC, None)
	entry.pop(ENT_FEATURES, None)

	# Transform complex dictionaries into lists (only where there are no any values)- I checked the values of those subentries
	entry[ENT_OBJ_ATTRIBUTE] = list(entry.get(ENT_OBJ_ATTRIBUTE, {}).keys())
	entry[ENT_SUBJ_ATTRIBUTE] = list(entry.get(ENT_SUBJ_ATTRIBUTE, {}).keys())
	entry[ENT_IND_OBJ_ATTRIBUTE] = list(entry.get(ENT_IND_OBJ_ATTRIBUTE, {}).keys())
	entry[ENT_NOUN] = list(entry.get(ENT_NOUN, {}).keys())
	entry[ENT_SEMI_AUTOMATIC] = entry.get(ENT_SEMI_AUTOMATIC, False)

	# Add plural form if no such is given
	if ENT_PLURAL not in entry:
		entry[ENT_PLURAL] = engine.plural(entry[ENT_ORTH])

	# Avoiding NONE and T values as strings
	for subentry in deepcopy(entry).keys():
		if type(entry[subentry]) == str:
			if entry[subentry] == "T":
				entry[subentry] = True
			elif entry[subentry] in NONE_VALUES:
				entry[subentry] = None

		if not is_known(subentry, ["ENT"], "ENT"):
			entry.pop(subentry)

		for noun_property in entry[ENT_NOUN]:
			if not is_known(noun_property, ["NOUN"], "NOUN"):
				entry.pop(noun_property)

	# Extract some constraints for the nominalization as a whole
	det_poss_no_other_obj = list(entry.get(ARG_CONSTRAINT_DET_POSS_NO_OTHER_OBJ, {}).keys())
	n_n_mod_no_other_obj = list(entry.get(ARG_CONSTRAINT_N_N_MOD_NO_OTHER_OBJ, {}).keys())
	entry.pop(ARG_CONSTRAINT_DET_POSS_NO_OTHER_OBJ, None)
	entry.pop(ARG_CONSTRAINT_N_N_MOD_NO_OTHER_OBJ, None)

	# By now, there can be only one nom-type for the nominalization
	nom_type = list(entry[ENT_NOM_TYPE].keys())[0]

	# Rearrange the nom-type properties
	entry[ENT_NOM_TYPE] = {TYPE_OF_NOM: nom_type,
						   TYPE_PART: entry[ENT_NOM_TYPE][nom_type].get(OLD_COMP_ADVAL, []),
						   TYPE_PP: entry[ENT_NOM_TYPE][nom_type].get(OLD_COMP_PVAL, [])}

	# A nominalization that get particle must get exactly one possible particle
	if "PART" in nom_type and len(entry[ENT_NOM_TYPE][TYPE_PART]) != 1:
		raise Exception(f"One particle should be specified for PART-typed nominalization ({get_current_specs()}).")

	subcats = entry.get(ENT_VERB_SUBC, {}).values()
	any_alternates_appear = any([SUBCAT_CONSTRAINT_ALTERNATES in subcat.keys() or OLD_SUBCAT_CONSTRAINT_ALTERNATES_OPT in subcat.keys() for subcat in subcats])

	# Correct typos in subcategorization types
	for subcat_type in deepcopy(entry).get(ENT_VERB_SUBC, {}).keys():
		# Check the current subcat type
		subcat_type = check_subcat_type(entry, subcat_type)

		# Ignore unknown subcats
		if subcat_type is None:
			continue

		subcat_info = entry[ENT_VERB_SUBC][subcat_type]

		# Specify the no other object constraints for any subcategorization (Relevant for the NOMLEX-2001 only)
		subcat_info[ARG_CONSTRAINT_DET_POSS_NO_OTHER_OBJ] = det_poss_no_other_obj
		subcat_info[ARG_CONSTRAINT_N_N_MOD_NO_OTHER_OBJ] = n_n_mod_no_other_obj

		# Make sure that the particle is specified in any subcat that should get a particle
		if "PART" in nom_type and "PART" in subcat_type:
			subcat_info[OLD_COMP_ADVAL] = list(set(subcat_info.get(OLD_COMP_ADVAL, []) + entry[ENT_NOM_TYPE][TYPE_PART]))

		# Add the ALTERNATES constraint for any subcat if the entry don't contain any ALTERNATES tag
		# Tags that were written in illegal subcats will be remove afterwards
		if not any_alternates_appear:
			if FEATURE_SUBJ_OBJ_ALT in nom_features or FEATURE_SUBJ_IND_OBJ_ALT in nom_features:
				entry[ENT_VERB_SUBC][subcat_type][SUBCAT_CONSTRAINT_ALTERNATES] = "T"

		# Remove ALTERNATES tag from illegal subcats (based on the features)
		if SUBCAT_CONSTRAINT_ALTERNATES in subcat_info.keys():
			# SUBJ-OBJ-ALT can only appear for intransitive subcats (intransitive -> transitive)
			if FEATURE_SUBJ_OBJ_ALT in nom_features and get_verb_type(subcat_type) != VERB_TYPE_INTRANS:
				entry[ENT_VERB_SUBC][subcat_type].pop(SUBCAT_CONSTRAINT_ALTERNATES)

			# SUBJ-IND-OBJ-ALT can only appear for transitive subcats (transitive -> ditransitive)
			elif FEATURE_SUBJ_IND_OBJ_ALT in nom_features and get_verb_type(subcat_type) != VERB_TYPE_TRANS:
				entry[ENT_VERB_SUBC][subcat_type].pop(SUBCAT_CONSTRAINT_ALTERNATES)

			elif FEATURE_SUBJ_OBJ_ALT in nom_features:
				alt_subcats[FEATURE_SUBJ_OBJ_ALT].append(subcat_type)
			elif FEATURE_SUBJ_IND_OBJ_ALT in nom_features:
				alt_subcats[FEATURE_SUBJ_IND_OBJ_ALT].append(subcat_type)



def rearrange_entry(entry):
	"""
	Rearranges the given entry
	:param entry: an entry in the lexicon as a dictionary of dictionaries ({VERB: ..., ORTH: ..., VERB-SUBC: {...}})
	:return: three updated entries (verb's entry, nominalization's entry, plural's entry)
	"""

	rearrange_entry_properties(entry)

	verb_entry = deepcopy(entry)
	verb_entry[ENT_VERB_SUBC] = {}
	nom_entry = deepcopy(entry)
	nom_entry[ENT_VERB_SUBC] = {}

	# By now, there can be only one nom-type for the nominalization
	nom_type = entry[ENT_NOM_TYPE][TYPE_OF_NOM]
	does_nom_have_part = "PART" in nom_type

	# Generate subcat version for nominalization and verb
	for subcat_type, subcat_info in entry.get(ENT_VERB_SUBC, {}).items():
		curr_specs["subcat"] = subcat_type

		# Split the nominalization and verb subentries for that subcat
		verb_subcat, nom_subcat = split_nom_and_verb_subcat(entry, subcat_info, subcat_type)

		# Assumption- PART subcats relates for verbs or noms that incorporate a particle
		if (not does_nom_have_part and "PART" not in subcat_type) or does_nom_have_part:
			nom_entry[ENT_VERB_SUBC][subcat_type] = nom_subcat

		verb_entry[ENT_VERB_SUBC][subcat_type] = verb_subcat

	nom_entry.pop(ENT_VERB_SUBJ, None)

	# Handling the verb of the nominalization (if given in the lexicon)
	if ENT_VERB in verb_entry.keys():
		verb_entry[ENT_ORTH] = verb_entry[ENT_VERB]
		verb_entry.pop(ENT_VERB_SUBJ, None)
		verb_entry.pop(ENT_NOM_TYPE, None)
		verb_entry.pop(ENT_VERB, None)
		verb_entry.pop(ENT_NOUN, None)
		verb_entry.pop(ENT_PLURAL, None)
		verb_entry.pop(ENT_PLURAL_FREQ, None)
		verb_entry.pop(ENT_SINGULAR_FALSE, None)
	else:
		verb_entry = None

	# Handling the plural of the nominalization (if existed)
	if nom_entry.get(ENT_PLURAL, None) is not None: # Check for NONE or *NONE*
		plural_entry = deepcopy(nom_entry)
		plural_entry[ENT_ORTH] = plural_entry[ENT_PLURAL]
		plural_entry.pop(ENT_PLURAL, None)
		nom_entry.pop(ENT_PLURAL_FREQ, None)
	else:
		plural_entry = None

	# If the nominalization entry can't be written in singular form
	if nom_entry.get(ENT_SINGULAR_FALSE, False):
		nom_entry = None

	return verb_entry, nom_entry, plural_entry