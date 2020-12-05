from copy import deepcopy

from .lexicon_modifications import lexicon_fixes_dict, nom_types_to_args_dict
from .utils import get_right_value, get_current_specs, without_part, curr_specs, get_verb_type
from noun_as_verb.utils import difference_list
from noun_as_verb.constants.lexicon_constants import *

# For debug
nom_roles_for_pval = set()



def transform_to_list(complement):
	"""
	Simplies a given options of complement and arranges them as a list
	:param complement: an information about the possible options of an argument\complement (as a list, dictionary or string (*MONE*))
	:return: an arranged list containing the leftout options for the given complement
	"""

	positions = []
	tmp_complement = deepcopy(complement)

	# Getting an initial list of options
	if type(tmp_complement) == list:
		positions = tmp_complement
	elif type(tmp_complement) == str:
		if tmp_complement in NONE_VALUES: # check for *NONE* or NONE
			positions = [tmp_complement]
		else:
			raise Exception(f"Illegal complement value ({get_current_specs()}).")

	else: # dictionary
		# Avoiding PP sub-entry
		if COMP_PP in tmp_complement.keys():
			positions += tmp_complement[COMP_PP].get(OLD_COMP_PVAL, [])
			del tmp_complement[COMP_PP]

		# Avoiding IND-OBJ-OTHER sub-entry (for more positions of IND-OBJ)
		if "IND-OBJ-OTHER" in tmp_complement.keys():
			positions += tmp_complement["IND-OBJ-OTHER"][OLD_COMP_PVAL]
			del tmp_complement["IND-OBJ-OTHER"]

		# Treating all the leftout keys as additional positions
		positions = positions + list(tmp_complement.keys())

	# Replacing directional prepositions with a fixed list
	if P_DIR in positions:
		positions.remove(P_DIR)
		positions += P_DIR_REPLACEMENTS

	# Replacing locational prepositions with a fixed list
	if P_LOC in positions:
		positions.remove(P_LOC)
		positions += P_LOC_REPLACEMENTS

	# Removing PP- prefixes for prepositions, and NONE values
	for positions_idx in range(len(positions)):
		if positions[positions_idx].startswith("PP-"):
			positions[positions_idx] = positions[positions_idx].replace("PP-", "").lower()

		if positions[positions_idx] in NONE_VALUES: # check for *NONE* or NONE
			positions[positions_idx] = OPT_POS

	return list(set(positions))

def rearrange_subject(subcat, default_subj_positions):
	"""
	Rearranges the possible positions for the subject argument [avoiding defaults- "by" unless "NOT-PP-BY", default subject]
	:param subcat: a dictionary of the subcategorization info
	:param default_subj_positions: a dictionary of the default subject positions (when the subcat don't include any subject positions)
	:return: None
	"""

	# Adding the default subject list in case it wasn't over-written
	if COMP_SUBJ not in subcat.keys() and default_subj_positions != {}:
		subcat[COMP_SUBJ] = deepcopy(default_subj_positions)

	# Updating the subject entry to be a list of possible positions
	if COMP_SUBJ in subcat.keys():
		subcat[COMP_SUBJ] = transform_to_list(subcat[COMP_SUBJ])

		# Avoiding the NOT-PP-BY tag
		if "NOT-PP-BY" in subcat[COMP_SUBJ]:
			subcat[COMP_SUBJ].remove("NOT-PP-BY")

		# Avoiding the BY preposition default for subjects
		elif "by" not in subcat[COMP_SUBJ]:
			subcat[COMP_SUBJ].append("by")

def rearrange_object(subcat):
	"""
	Rearranges the possible possitions for the object argument
	:param subcat: a dictionary of the subcategorization info
	:return: None
	"""

	if COMP_OBJ in subcat.keys():
		# Updating the object entry to be a list of possible positions
		subcat[COMP_OBJ] = transform_to_list(subcat[COMP_OBJ])

def rearrange_ind_object(subcat):
	"""
	Rearranges the possible positions for the subject argument [avoiding long named values- "IND-OBJ-..." and "IND-OBJ-OTHER"]
	:param subcat: a dictionary of the subcategorization info
	:return: None
	"""

	if COMP_IND_OBJ in subcat.keys():
		# Updating the indirect-object entry to be a list of possible positions
		subcat[COMP_IND_OBJ] = transform_to_list(subcat[COMP_IND_OBJ])

		# Removing IND-OBJ- position tags to normal preposition tags
		for position_idx in range(len(subcat[COMP_IND_OBJ])):
			curr_position = subcat[COMP_IND_OBJ][position_idx]
			if curr_position.startswith(COMP_IND_OBJ + "-"):
				subcat[COMP_IND_OBJ][position_idx] = curr_position.replace(COMP_IND_OBJ + "-", "").lower()

def rearrange_preps(subcat, subcat_type, is_verb=False):
	"""
	Rearranges the positions of the prepositions that appear in the given subcat dictionary [replacing old representations and duplicating missing representations]
	:param subcat: a dictionary of the subcategorization info
	:param subcat_type: the type of the subcategorization (for determing if the number of needed prepositions)
	:param is_verb: whether or not the given subcat is for verb rearranging (otherwise- nominalization)
	:return: None
	"""
	global nom_roles_for_pval

	# Translating the names of the prepositional complements (from PVAL* to PP*)
	if OLD_COMP_PVAL in subcat.keys():
		subcat[COMP_PP] = transform_to_list(subcat[OLD_COMP_PVAL])
	if OLD_COMP_PVAL1 in subcat.keys():
		subcat[COMP_PP1] = transform_to_list(subcat[OLD_COMP_PVAL1])
	if OLD_COMP_PVAL2 in subcat.keys():
		subcat[COMP_PP2] = transform_to_list(subcat[OLD_COMP_PVAL2])

	nom_roles_for_pval.update(transform_to_list(subcat.get("NOM-ROLE-FOR-PVAL", {})) + transform_to_list(subcat.get("NOM-ROLE-FOR-PVAL1", {})) + transform_to_list(subcat.get("NOM-ROLE-FOR-PVAL2", {})))

	# Replacing the NOM-ROLE-FOR-PVAL* and PVAL-NOM* subentries by adding more options to PP*, only for nominalizations
	# PVAL-NOM* overwrites PVAL positions and NOM-ROLE-FOR-PVAL gives more options for PVAL*
	if not is_verb:
		if "NOM-ROLE-FOR-PVAL" in subcat.keys():
			subcat[COMP_PP] = subcat.get(OLD_COMP_PVAL_NOM, subcat.get(COMP_PP, []))
			roles_for_pval = transform_to_list(subcat["NOM-ROLE-FOR-PVAL"])
			subcat[COMP_PP] += roles_for_pval
		if "NOM-ROLE-FOR-PVAL1" in subcat.keys():
			subcat[COMP_PP1] = subcat.get(OLD_COMP_PVAL_NOM1, subcat.get(COMP_PP1, []))
			subcat[COMP_PP1] += transform_to_list(subcat["NOM-ROLE-FOR-PVAL1"])
		if "NOM-ROLE-FOR-PVAL2" in subcat.keys():
			subcat[COMP_PP2] = subcat.get(OLD_COMP_PVAL_NOM2, subcat.get(COMP_PP2, []))
			subcat[COMP_PP2] += transform_to_list(subcat["NOM-ROLE-FOR-PVAL2"])

	# Removing the unnecessary complements from the old representation of the lexicon
	subcat.pop(OLD_COMP_PVAL, None)
	subcat.pop(OLD_COMP_PVAL_NOM, None)
	subcat.pop("NOM-ROLE-FOR-PVAL", None)
	subcat.pop(OLD_COMP_PVAL1, None)
	subcat.pop(OLD_COMP_PVAL_NOM1, None)
	subcat.pop("NOM-ROLE-FOR-PVAL1", None)
	subcat.pop(OLD_COMP_PVAL2, None)
	subcat.pop(OLD_COMP_PVAL_NOM2, None)
	subcat.pop("NOM-ROLE-FOR-PVAL2", None)

	# Fixing the lack of one preposition, when two prepositions are needed (for NOM-PP-PP and NOM-NP-PP-PP subcats)
	# The positions of PP2 are assumed to be the same as those of PP1 or PP
	if "PP-PP" in subcat_type:
		if COMP_PP2 not in subcat.keys():
			if COMP_PP1 in subcat.keys():
				subcat[COMP_PP2] = subcat[COMP_PP1]
			elif COMP_PP in subcat.keys():
				subcat[COMP_PP2] = subcat[COMP_PP]
			else:
				raise Exception(f"A subcat that requires two prepositions, don't give possible position for none of them ({get_current_specs()}).")

def rearrange_not(subcat, is_verb=False):
	"""
	Rearranges the constraints that in the NOT subentry [replacing old representations]
	:param subcat: a dictionary of the subcategorization info
	:param is_verb: whether or not the given subcat is for verb rearranging (otherwise- nominalization)
	:return: None
	"""

	if SUBCAT_NOT in subcat.keys():
		# Aggregating a new subentry for NOT
		new_not = []
		for and_not_name, and_not_value in subcat[SUBCAT_NOT].items():
			new_and_not = {}

			# Rearranging the prepositions in the AND subentry under NOT
			rearrange_preps(and_not_value, "", is_verb)

			for argument_name, argument_value in and_not_value.items():
				new_argumnet_value = transform_to_list(argument_value)

				# For verbs, keep constraints only concerning specific complements
				if not is_verb or argument_name not in [COMP_SUBJ, COMP_OBJ]:
					new_and_not[argument_name] = new_argumnet_value

			# Add not constraints with more than one condition
			if len(new_and_not.keys()) > 1:
				new_not.append(new_and_not)

		# Add the new NOT subentry only if not empty
		if new_not != []:
			subcat[SUBCAT_NOT] = new_not
		else:
			del subcat[SUBCAT_NOT]

def rearrange_requires_and_optionals(subcat, subcat_type, default_requires, other_subcat_types, is_verb=False):
	"""
	Rearranges the requires and optionals for the given subcat entry
	:param subcat: a dictionary of the subcategorization info
	:param subcat_type: the type of the subcategorization (for determing whethet the object is required)
	:param default_requires: list of arguments that and constraints that are required for the given subcat
	:param other_subcat_types: list of the other subcat types in the current lexicon entry
	:param is_verb: whether or not the given subcat is for verb rearranging (otherwise- nominalization)
	:return: None
	"""

	requires = []
	optionals = []

	if is_verb and COMP_PART in subcat:
		requires.append(COMP_PART)

	# Updating the list of required complements
	for complement_type in subcat.get(SUBCAT_REQUIRED, {}).keys():
		curr_specs["comp"] = complement_type

		# The required arguments are the ones with no constraints in the required subentry
		if list(subcat[SUBCAT_REQUIRED][complement_type].keys()) == []:
			requires.append(complement_type)
		else:
			# Otherwise, the arguments are required under one of the constraints: DET-POSS-ONLY or N-N-MOD-ONLY
			# Specify those constraints for the subcategorization (Relevant for the NOMLEX-plus only)

			if "DET-POSS-ONLY" in list(subcat[SUBCAT_REQUIRED][complement_type].keys()):
				subcat[ARG_CONSTRAINT_DET_POSS_NO_OTHER_OBJ] += [complement_type]

			if "N-N-MOD-ONLY" in list(subcat[SUBCAT_REQUIRED][complement_type].keys()):
				subcat[ARG_CONSTRAINT_N_N_MOD_NO_OTHER_OBJ] += [complement_type]

	# Adding complements with possible optional positon to the optional list
	tmp_subcat = deepcopy(subcat)
	for complement_type in difference_list(tmp_subcat.keys(), [ARG_CONSTRAINT_DET_POSS_NO_OTHER_OBJ, ARG_CONSTRAINT_N_N_MOD_NO_OTHER_OBJ]):
		curr_specs["comp"] = complement_type

		if OPT_POS in subcat[complement_type]:
			optionals += [complement_type]
			subcat[complement_type].remove(OPT_POS)

			# Assumption- OPTIONAL-POSITION value (meaning NONE\*NONE*) is preferable to the information in the requires list
			if complement_type in requires:
				requires.remove(complement_type)

		# Delete the complement if it has no possible options
		if subcat[complement_type] == []:
			del subcat[complement_type]

	curr_specs["comp"] = None

	# All the non-optional constraints in the default requires list are also required
	requires += difference_list(default_requires, optionals)

	# OBJECT is optional for NOM-NP-X subcats, only if NOM-X isn't compatible with the current entry, otherwise it is required
	if without_part(subcat_type).startswith("NOM-NP"):
		# Object is required in the next cases
		obj_is_required = False
		if subcat_type == "NOM-NP":
			if not {"NOM-INTRANS", "NOM-INTRANS-RECIP"}.isdisjoint(other_subcat_types):
				obj_is_required = True
		elif subcat_type == "NOM-NP-AS-NP-SC":
			if "NOM-AS-NP" in other_subcat_types:
				obj_is_required = True
		else:
			subcat_without_np = subcat_type.replace("NOM-PART-NP", "NOM-PART")
			subcat_without_np = subcat_without_np.replace("NOM-NP-", "NOM-")
			if subcat_without_np in other_subcat_types:
				obj_is_required = True

		if obj_is_required:
			requires.append(COMP_OBJ)
			optionals = difference_list(optionals, [COMP_OBJ])
		elif COMP_OBJ not in requires:
			optionals.append(COMP_OBJ)

	# SUBJECT is optional by default
	if COMP_SUBJ not in requires:
		optionals.append(COMP_SUBJ)

	subcat[SUBCAT_REQUIRED] = list(set(requires))
	subcat[SUBCAT_OPTIONAL] = list(set(optionals))



def change_types(subcat, types_dict):
	"""
	Translates the types of the complements according to the types dictionary
	:param subcat: a dictionary of the subcategorization info
	:param types_dict: a ditionary of types ({old_type: new_type})
	:return: None
	"""

	# Moving over all the needed to be changed arguments
	for complement_type, new_complement_type in types_dict.items():
		curr_specs["comp"] = complement_type

		if complement_type == new_complement_type:
			continue

		# If the argument appears in the subcat info, change its name
		if complement_type in subcat.keys():
			if new_complement_type != IGNORE_COMP:
				subcat[new_complement_type] = deepcopy(subcat[complement_type])

				# Replacing the complement type on the required list
				if complement_type in difference_list(subcat[SUBCAT_REQUIRED], types_dict.values()):
					subcat[SUBCAT_REQUIRED].remove(complement_type)

					if new_complement_type not in subcat[SUBCAT_OPTIONAL]:
						subcat[SUBCAT_REQUIRED].append(new_complement_type)

				# Replacing the complement type on the optionals list
				if complement_type in difference_list(subcat[SUBCAT_OPTIONAL], types_dict.values()):
					subcat[SUBCAT_OPTIONAL].remove(complement_type)

					if new_complement_type not in subcat[SUBCAT_REQUIRED]:
						subcat[SUBCAT_OPTIONAL].append(new_complement_type)

			del subcat[complement_type]

	subcat[SUBCAT_REQUIRED] = list(set(subcat[SUBCAT_REQUIRED]))
	subcat[SUBCAT_OPTIONAL] = list(set(subcat[SUBCAT_OPTIONAL]))
	curr_specs["comp"] = None

def get_special_values(entry, location_list):
	"""
	Returns the special values at the given location in the given lexicon entry (usually under NOM-SUBC)
	The function finds those values by recursive search (by calling itself)
	:param entry: an entry in the lexicon as a dictionary of dictionaries
	:param location_list: a list of sub-entries which lead to the list of values
	:return: the wanted special values
	"""

	# If the current entry is a list, then we have found the wanted values
	if type(entry) == list:
		return entry

	# Otherwise, keep looking recursively

	# Location that ends with "-" relates to any subentry that ends with "-' (like "P-" for P-POSSING, P-ING)
	if location_list[0].endswith("-"):
		values = []
		for subentry in entry.keys():
			if subentry.startswith(location_list[0]):
				values += get_special_values(entry[subentry], location_list[1:])

		return values

	# Otherwise, a specific subentry is the next
	if location_list[0] in entry.keys():
		return get_special_values(entry[location_list[0]], location_list[1:])

	return []

def use_special_case(subcat, special_cases_dict):
	"""
	Updates the positions for arguments based on non-standard locations in the subcat (like NOM-SUBC)
	:param subcat: a dictionary of the subcategorization info
	:param special_cases_dict: a dictionary with locations for special values in the lexicon
	:return: None
	"""

	# Updating missing arguments values according to the given locaitions dictionary
	for complement_type, location_list in special_cases_dict.items():
		curr_specs["comp"] = complement_type

		if complement_type not in subcat.keys():
			special_values = get_special_values(subcat, location_list)

			if special_values != []:
				subcat[complement_type] = special_values

	curr_specs["comp"] = None

	# After using the special cases, we can delete the "NOM-SUBC" subentry
	subcat.pop("NOM-SUBC", None)

def use_defaults(subcat, defaults_dict):
	"""
	Uses the defaults dictionary to update missing values for missing arguments
	:param subcat: a dictionary of the subcategorization info
	:param defaults_dict: a dictionary with lists of positions for specific arguments ({ARG: []}) suitable for the given subcat
	:return: None
	"""

	# Update missing arguments values according to the given default values
	for complement_type, default_positions in defaults_dict.items():
		curr_specs["comp"] = complement_type

		# Handle optional position differently
		if OPT_POS in default_positions:
			default_positions.remove(OPT_POS)

			# Add the optional complement to the optionals list, if that complement isn't required
			if complement_type not in subcat[SUBCAT_REQUIRED]:
				subcat[SUBCAT_OPTIONAL] += [complement_type]
				subcat[SUBCAT_OPTIONAL] = list(set(subcat[SUBCAT_OPTIONAL]))
			else:
				raise Exception(f"An optional complement according to a manual table is also supposed to be required ({get_current_specs()}).")

		if complement_type not in subcat.keys():
			if default_positions != []:
				subcat[complement_type] = default_positions

	curr_specs["comp"] = None

def add_extensions(subcat, is_verb=False):
	"""
	Extends the positions for certain compelements that gets a constant words after prefixes
	For example, P-WH-S that can get "whether" after a preposition (like "of")
	The extension is relevant only for prepositions
	:param subcat: a dictionary of the subcategorization info
	:param is_verb: whether or not the given subcat is for verb rearranging (otherwise- nominalization)
	:return: None
	"""

	get_options_with_extensions = lambda option, extensions: [option + " " + addition if option not in extensions else option for addition in extensions]

	# Adding additional fixed info for specific complements
	for complement_type in difference_list(subcat.keys(), [SUBCAT_CONSTRAINTS, SUBCAT_REQUIRED, SUBCAT_OPTIONAL, SUBCAT_NOT]):
		curr_specs["comp"] = complement_type

		# only for complements that are represented as a list (excluding NOT for example)
		if type(subcat[complement_type]) == list:
			new_options = []

			# Update each option for the complement, by adding constants after the prepositions
			for option in subcat[complement_type]:
				if complement_type in [COMP_WH_S, COMP_P_WH_S]:
					if is_verb:
						extensions = WH_VERB_OPTIONS
					else:
						extensions = WH_NOM_OPTIONS
				elif complement_type == COMP_WHERE_WHEN_S:
					extensions = WHERE_WHEN_OPTIONS
				elif complement_type == COMP_HOW_S:
					extensions = HOW_OPTIONS
				elif complement_type == COMP_HOW_TO_INF:
					extensions = HOW_TO_OPTIONS
				else:
					extensions = [option]

				if option in extensions:
					extensions = [option]

				new_options += get_options_with_extensions(option, extensions)

			subcat[complement_type] = new_options

	curr_specs["comp"] = None

def use_nom_type(subcat, nom_type_info, is_verb=False):
	"""
	Uses the type of the nominalization and specifies it for the given subcat (in the relevant complement as NOM)
	Sometimes the type of the nom specifies the position of the nom for the verb, and this info should also be included for the verb only
	:param subcat: a dictionary of the subcategorization info
	:param nom_type_info: the type of the nominalization (as a dictionary)
	:param is_verb: whether or not the given subcat is for verb rearranging (otherwise- nominalization)
	:return: None
	"""

	# Get the type of complements that appropriate to the given type of nominalization
	type_of_nom = without_part(nom_type_info[TYPE_OF_NOM])
	complement_types = nom_types_to_args_dict.get(type_of_nom, [])

	changed = False

	# Search for the first appropriate complement that the subcat includes
	for complement_type in complement_types:
		curr_specs["comp"] = complement_type

		# For verbs, a relevant complement also gets the position of the nom as a new possible position
		if is_verb:
			if complement_type in subcat[SUBCAT_REQUIRED] + subcat[SUBCAT_OPTIONAL] and SUBCAT_CONSTRAINT_ALTERNATES not in subcat:
				if nom_type_info[TYPE_PP] != []:
					subcat[type_of_nom] = list(set(subcat.get(complement_type, []) + nom_type_info[TYPE_PP]))

					if type_of_nom != complement_type:
						subcat.pop(complement_type, None)

						if complement_type in subcat[SUBCAT_REQUIRED]: subcat[SUBCAT_REQUIRED] = list(set(subcat[SUBCAT_REQUIRED] + [type_of_nom]))
						else: subcat[SUBCAT_OPTIONAL] = list(set(subcat[SUBCAT_OPTIONAL] + [type_of_nom]))

				changed = True

		# For noms, the only position of the complement is NOM
		# The complement should appear in the required or optional lists
		elif complement_type in list(subcat.keys()) + subcat[SUBCAT_REQUIRED] + subcat[SUBCAT_OPTIONAL]: # or complement_type == COMP_INSTRUMENT
			changed = True

			# Instead of the founded relevant complement, we will write the type of nom as a new complement
			subcat.pop(complement_type, None)
			subcat[type_of_nom] = [POS_NOM]
			subcat[SUBCAT_REQUIRED] = list(set(subcat[SUBCAT_REQUIRED] + [type_of_nom])) # NOM must be required for the nominalization
			subcat[SUBCAT_OPTIONAL] = difference_list(subcat[SUBCAT_OPTIONAL], [type_of_nom])

		if changed:
			# Remove the old complement type from both required and optional lists
			# The founded type can be different than the searched on only for IND-OBJ
			if complement_type != type_of_nom:
				subcat[SUBCAT_REQUIRED] = list(set(difference_list(subcat[SUBCAT_REQUIRED], [complement_type])))
				subcat[SUBCAT_OPTIONAL] = list(set(difference_list(subcat[SUBCAT_OPTIONAL], [complement_type])))

			# If we replaced PP1 with IND-OBJ, then PP2 should actually mean the complement PP
			if complement_type == COMP_PP1:
				subcat[COMP_PP] = difference_list(subcat.pop(COMP_PP2), [POS_NOM])  # PP2 cannot also be the NOM
				if COMP_PP2 in subcat[SUBCAT_REQUIRED]: subcat[SUBCAT_REQUIRED] = difference_list(subcat[SUBCAT_REQUIRED], [COMP_PP2]) + [COMP_PP]
				else: subcat[SUBCAT_OPTIONAL] = difference_list(subcat[SUBCAT_OPTIONAL], [COMP_PP2]) + [COMP_PP]

			break

	curr_specs["comp"] = None

def perform_alternation(subcat, subcat_type):
	"""
	Performs alternation over the given subcategorization, when the ALTERNATES tag appears
	:param subcat: a dictionary of the subcategorization info
	:param subcat_type: the type of the subcategorization (for determing if the number of needed prepositions)
	:return: None
	"""

	# Check that this subcat is tagged with alteration constraint
	if subcat.get(SUBCAT_CONSTRAINT_ALTERNATES, "F") != "T":
		return

	# SUBJ-IND-OBJ-ALT (transitive -> ditransitive)
	if get_verb_type(subcat_type) == VERB_TYPE_TRANS:
		if COMP_IND_OBJ in subcat.keys() or COMP_SUBJ not in subcat.keys():
			raise Exception(f"A conflict of SUBJ-IND-OBJ-ALT feature- OBJECT must appear and IND-OBJECT must not ({get_current_specs()}).")

		replace_complement = COMP_SUBJ
		new_complement = COMP_IND_OBJ

	elif get_verb_type(subcat_type) == VERB_TYPE_INTRANS: # SUBJ-OBJ-ALT (intransitive -> transitive)
		if COMP_OBJ in subcat.keys() or COMP_SUBJ not in subcat.keys():
			raise Exception(f"A conflict of SUBJ-OBJ-ALT feature- SUBJECT must appear and OBJECT must not ({get_current_specs()}).")

		replace_complement = COMP_SUBJ
		new_complement = COMP_OBJ
	else:
		raise Exception(f"Illegal subcat-type for alternation ({get_current_specs()}).")

	# Avoid alternation in case the argument can be the nominalization
	# In such cases (like renter or boiler) the nom has a purpose and ALTERNATES don't affect it
	# It only affects the suitable verb
	if POS_NOM in subcat[replace_complement]:
		return

	# Performing the alternation
	subcat[new_complement] = deepcopy(subcat[replace_complement])
	del subcat[replace_complement]

	# Replacing the required or optionality of the alternated complements
	if replace_complement in subcat[SUBCAT_REQUIRED]:
		subcat[SUBCAT_REQUIRED].append(new_complement)
		subcat[SUBCAT_REQUIRED].remove(replace_complement)
	else:
		subcat[SUBCAT_OPTIONAL].append(new_complement)
		subcat[SUBCAT_OPTIONAL].remove(replace_complement)

	# The replaced argument is still part of the structure of the subcat
	# Thus it can be appeared in other position (like NOM which is changed afterwards)
	subcat[SUBCAT_OPTIONAL].append(replace_complement)



def simplify_subcat(entry, subcat, subcat_type, is_verb=False):
	"""
	Simplifies the given subcat dictionary, in a way that makes it more consistent
	The new precedure traslates any argument positions dictionary into a list, and fixes some mistakes and missing values in the nominalization
	:param entry: an entry in the lexicon as a dictionary of dictionaries
	:param subcat: a dictionary of the subcategorization info ({ARG1: {POS1: {...}, POS2: {...}, ...}, ARG2: {...}, NOT: {...}, REQUIRED: {...}, OPTIONALS: {...}})
	:param subcat_type: the type of the subcategorization
	:param is_verb: whether or not the given subcat is for verb rearranging (otherwise- nominalization)
	:return: None
	"""

	# Update the subject and the object differently for verbs and noms
	if is_verb:
		subcat[COMP_SUBJ] = [POS_NSUBJ, POS_DET_POSS, "by"]

		# Object is relevant only for transitive verbs
		if without_part(subcat_type).startswith("NOM-NP"):
			subcat[COMP_OBJ] = [POS_DOBJ, POS_NSUBJPASS]

		# Does the verb requires a particle?
		if OLD_COMP_ADVAL in subcat:
			subcat[COMP_PART] = subcat.pop(OLD_COMP_ADVAL) + [POS_PART]
	else:
		rearrange_subject(subcat, entry.get(ENT_VERB_SUBJ, {}))
		rearrange_object(subcat)
		subcat.pop(COMP_PART, None)
		subcat.pop(OLD_COMP_ADVAL, None)

	rearrange_ind_object(subcat)

	# Get the related information from the constant list of fixes for the lexicon
	requires, defaults, names, special_cases = get_right_value(lexicon_fixes_dict, subcat_type, is_verb=is_verb)

	# Rearranging the subcategorization- the order matters
	rearrange_preps(subcat, subcat_type, is_verb=is_verb)
	rearrange_requires_and_optionals(subcat, subcat_type, requires, entry[ENT_VERB_SUBC].keys(), is_verb=is_verb)
	change_types(subcat, names)
	use_special_case(subcat, special_cases)
	use_defaults(subcat, defaults)
	add_extensions(subcat, is_verb=is_verb)
	rearrange_not(subcat, is_verb=is_verb)
	use_nom_type(subcat, entry[ENT_NOM_TYPE], is_verb=is_verb)
	perform_alternation(subcat, subcat_type)