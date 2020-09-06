import re
import numpy as np
from copy import deepcopy
from collections import defaultdict

from tqdm import tqdm

from arguments_extractor.lisp_to_json.simplify_lexicon_keys import remove_entries, split_entries
from arguments_extractor.lisp_to_json.simplify_entry import rearrange_entry, alt_subcats
from arguments_extractor.lisp_to_json.simplify_subcat import nom_roles_for_pval
from arguments_extractor.lisp_to_json.simplify_representation import argument_constraints, missing_required, args_without_pos
from arguments_extractor.lisp_to_json.utils import get_current_specs, curr_specs, is_known, without_part, get_right_value, unknown_values_dict, known_values_dict
from arguments_extractor.constants.lexicon_constants import *
from arguments_extractor.utils import difference_list, list_to_regex
from arguments_extractor import config

# For debug
added_noms = []
removed_noms = []
noms_with_missing_positions = []  # DET-POSS or N-N-MOD are missing
argument_properties = defaultdict(set)
entries_by_type = defaultdict(set)
complements_per_subcat = defaultdict(set)
arguments_collisions = defaultdict(list)


def sanity_checks(lexicon, is_verb=False):
	curr_specs["is_verb"] = is_verb

	for word in lexicon.keys():
		word_entry = lexicon[word]
		curr_specs["word"] = word_entry[ENT_ORTH]

		if not is_verb: is_known(without_part(word_entry[ENT_NOM_TYPE][TYPE_OF_NOM]), ["NOM_TYPE"], "NOM-TYPE")

		for subentry in lexicon[word].keys():
			if lexicon[word][subentry] is not None:
				entries_by_type[type(lexicon[word][subentry])].update([subentry])

		for subcat_type, subcat in lexicon[word][ENT_VERB_SUBC].items():
			curr_specs["subcat"] = subcat_type
			optionals = subcat[SUBCAT_OPTIONAL]
			requires = subcat[SUBCAT_REQUIRED]

			for constraint in subcat[SUBCAT_CONSTRAINTS]:
				is_known(constraint, ["SUBCAT_CONSTRAINT"], "SUBCAT COMPLEMENTS & CONSTRAINTS")

			if len(set(requires)) != len(requires):
				print(requires)
				raise Exception(f"Requires list isn't unique ({get_current_specs()}).")

			if len(set(optionals)) != len(optionals):
				print(optionals)
				raise Exception(f"Optionals list isn't unique ({get_current_specs()}).")

			# Check that the requires and the optionals lists aren't intersecting
			if set(difference_list(optionals, requires)) != set(optionals):
				print(requires)
				print(optionals)
				raise Exception(f"Requires and optionals are intersecting ({get_current_specs()}).")

			all_complements = difference_list(subcat.keys(), [SUBCAT_OPTIONAL, SUBCAT_REQUIRED, SUBCAT_NOT, SUBCAT_CONSTRAINTS])
			complements_per_subcat[subcat_type].update(all_complements)

			if not(set(optionals + requires) >= set(all_complements)):
				print(set(optionals + requires))
				print(set(all_complements))
				raise Exception(f"Some complements don't appear in required or optional ({get_current_specs()}).")

			# Check that all the required are specified in the subcategorization
			if difference_list(requires, subcat.keys()) != []:
				raise Exception(f"There is a required argument without a specification ({get_current_specs()}).")

			positions_per_complement = defaultdict(list)

			for complement_type in all_complements:
				curr_specs["comp"] = complement_type

				for linked_arg in subcat[complement_type]:
					positions_per_complement[complement_type] += subcat[complement_type][linked_arg][ARG_POSITIONS] + subcat[complement_type][linked_arg][ARG_PREFIXES]

					complement_info = subcat[complement_type][linked_arg]
					for constraint in complement_info[ARG_CONSTRAINTS]:
						is_known(constraint, ["ARG_CONSTRAINT"], "ARG CONSTRAINTS")

					if (ARG_CONSTRAINT_DET_POSS_NO_OTHER_OBJ in complement_info[ARG_CONSTRAINTS] and POS_DET_POSS not in complement_info[ARG_POSITIONS]) or \
					   (ARG_CONSTRAINT_N_N_MOD_NO_OTHER_OBJ in complement_info[ARG_CONSTRAINTS] and POS_N_N_MOD not in complement_info[ARG_POSITIONS]):
						noms_with_missing_positions.append(word)

					argument_properties[ARG_POSITIONS].update(complement_info[ARG_POSITIONS])
					argument_properties[ARG_PREFIXES].update(complement_info[ARG_PREFIXES])
					argument_properties[ARG_ILLEGAL_PREFIXES].update(complement_info.get(ARG_ILLEGAL_PREFIXES, []))

			for complement_type, positions in positions_per_complement.items():
				for other_complement_type, other_positions in positions_per_complement.items():
					pos_intersection = set(positions).intersection(other_positions)
					if complement_type != other_complement_type and len(pos_intersection) != 0 and pos_intersection != {POS_PREFIX}:
						collided_args = sorted(list({complement_type, other_complement_type}))
						arguments_collisions[tuple(collided_args)].append(word)

			curr_specs["comp"] = None
			more_argument_constraints = get_right_value(argument_constraints, subcat_type, {}, is_verb)

			for complement_type in more_argument_constraints.keys():
				curr_specs["comp"] = complement_type
				if complement_type not in subcat.keys():
					continue

				for linked_arg in subcat[complement_type]:
					auto_controlled = []

					# Automatic constraints
					if complement_type.endswith("-POC"):
						auto_controlled = [COMP_PP]
					elif complement_type.endswith("-NPC"):
						auto_controlled = [COMP_NP]
					elif complement_type.endswith("-OC"):
						auto_controlled = [COMP_OBJ]
					elif complement_type.endswith("-SC"):
						auto_controlled = [COMP_SUBJ]
					elif complement_type.endswith("-VC"):
						auto_controlled = [COMP_SUBJ, COMP_OBJ]

						if subcat_type == "NOM-P-NP-TO-INF-VC":
							auto_controlled = [COMP_SUBJ, COMP_PP]

					# Assure that the manual constraints were added correctly
					if set(auto_controlled) != set(subcat[complement_type][linked_arg].get(ARG_CONTROLLED, [])):
						print(subcat[complement_type][linked_arg].get(ARG_CONTROLLED, []))
						print(auto_controlled)
						raise Exception(f"Manual controlled constraints do not agree with the automatic ones ({get_current_specs()}).")

					if subcat[complement_type][linked_arg][ARG_POSITIONS] == []:
						print(word, subcat_type, complement_type)
						raise Exception(f"There is a complement without any position ({get_current_specs()}).")

			curr_specs["comp"] = None

def get_summary():
	print("\nValues that were found during the creation of the new representation:")
	print("----------------------")

	print(f"Possible nom roles for PVAL: {nom_roles_for_pval}")
	print(f"Required arguments that are sometimes missing: {list(zip(*np.unique(missing_required, return_counts=True)))}")
	print(f"Words with missing DET-POSS/N-N-MOD (based on NO-OTHER-OBJ feature): {noms_with_missing_positions}")
	print(f"Complements that don't specify specific postag: {set(args_without_pos)}")

	for constant_type in list(set(list(unknown_values_dict.keys()) + list(known_values_dict.keys()))):
		print(f"\n{constant_type}:")
		print("----------------------")
		print("RELEVANT:", known_values_dict.get(constant_type, {}))
		print("IGNORED:", unknown_values_dict.get(constant_type, {}))

	print("\nPossible positions by type:")
	print("----------------------")
	for argument_property, possible_values in argument_properties.items():
		print(f"{argument_property}: {possible_values}")

		if argument_property == "PREFIXES":
			only_preposition = lambda pos: re.sub(list_to_regex(WHERE_WHEN_OPTIONS + WH_VERB_OPTIONS + HOW_TO_OPTIONS + HOW_OPTIONS, "|"), '', pos).strip()
			print(f"MULTI-WORD {argument_property}: {set([only_preposition(pos) for pos in possible_values if len(only_preposition(pos).split(' ')) > 1])}")

	print("\nRelevant entries by type:")
	print("----------------------")
	for entries_type, entries_list in entries_by_type.items():
		print(f"{entries_type.__name__}: {entries_list}")

	print("\nColliding complement types:")
	print("----------------------")
	for collided_args, words in arguments_collisions.items():
		words = list(set(words))
		print(f"{collided_args}: {len(words)}, such as {words[:5]}")



def add_to_lexicon(lexicon, word_entry, word_orth=None):
	"""
	Adds a word entry to the given lexicon
	:param lexicon: a specific lexicon as dictionary ({WORD#: {ORTH: WORD, VERB-SUBC: {...}, ...}})
	:param word_entry: an entry that is needed to be added to the lexicon
	:param word_orth: the word that is needed to be added to the lexicon (optional)
	:return: the key of the word/s that were added to the lexicon (it can be list in some cases)
	"""

	if word_entry is None:
		return None

	if word_orth is None:
		word_orth = word_entry[ENT_ORTH]

	# If the given word is a list (like when there are more than one verb/plural that suitable a single sense of a nominalization)
	if type(word_orth) == list:
		list_of_orths = word_orth
		return [add_to_lexicon(lexicon, word_entry, word_orth) for word_orth in list_of_orths]

	# Cleaning the orth word from numbers
	if "#" in word_orth:
		word_orth = "".join(word_orth.split("#")[0:-1])

	# The feature ORTH will remain as the clean word_orth (without the numbers)
	word_entry[ENT_ORTH] = word_orth

	# The first orth appears without number
	if word_orth not in lexicon.keys():
		lexicon[word_orth] = word_entry
		return word_orth

	# If this orth already appears in the lexicon, then find the last one (using the NEXT feature)
	count_same_orth = 2
	curr_word_entry = lexicon[word_orth]
	while ENT_NEXT in curr_word_entry.keys():
		curr_word_entry = lexicon[curr_word_entry[ENT_NEXT]]
		count_same_orth += 1

	# Add the correct number to the current word_orth (based on the number of orth words that were found)
	numbered_word = f"{word_orth}#{count_same_orth}"
	curr_word_entry[ENT_NEXT] = numbered_word
	lexicon[numbered_word] = word_entry

	return numbered_word

def add_default_entry(lexicon):
	default_entry = {ENT_VERB_SUBC: {DEFAULT_SUBCAT: {}}}
	default_subcat = default_entry[ENT_VERB_SUBC][DEFAULT_SUBCAT]

	for word_entry in lexicon.values():
		for subcat in word_entry[ENT_VERB_SUBC].values():
			all_complements = difference_list(subcat.keys(), [SUBCAT_OPTIONAL, SUBCAT_REQUIRED, SUBCAT_NOT, SUBCAT_CONSTRAINTS])

			for complement_type in all_complements:
				complement_info = subcat[complement_type]
				clean_complement_type = re.sub("PP1|PP2", 'PP', complement_type)
				clean_complement_type = re.sub("-OC|-SC|-POC|-NPC|-VC", '', clean_complement_type)

				if clean_complement_type not in default_subcat:
					default_subcat[clean_complement_type] = {}

				for linked_arg in complement_info.keys():
					if linked_arg not in default_subcat[clean_complement_type]:
						default_subcat[clean_complement_type][linked_arg] = defaultdict(list)

					default_arg = default_subcat[clean_complement_type][linked_arg]

					default_arg[ARG_PREFIXES] += complement_info[linked_arg].get(ARG_PREFIXES, [])
					default_arg[ARG_POSITIONS] = list(set(default_arg[ARG_POSITIONS] + complement_info[linked_arg].get(ARG_POSITIONS, [])))
					default_arg[ARG_ROOT_UPOSTAGS] = list(set(default_arg[ARG_ROOT_UPOSTAGS] + complement_info[linked_arg].get(ARG_ROOT_UPOSTAGS, [])))
					default_arg[ARG_ROOT_URELATIONS] = list(set(default_arg[ARG_ROOT_URELATIONS] + complement_info[linked_arg].get(ARG_ROOT_URELATIONS, [])))
					default_arg[ARG_ROOT_PATTERNS] = list(set(default_arg[ARG_ROOT_PATTERNS] + complement_info[linked_arg].get(ARG_ROOT_PATTERNS, [])))

					constraints = difference_list(complement_info[linked_arg].get(ARG_CONSTRAINTS, []), [ARG_CONSTRAINT_PLURAL, ARG_CONSTRAINT_N_N_MOD_NO_OTHER_OBJ, ARG_CONSTRAINT_DET_POSS_NO_OTHER_OBJ])
					default_arg[ARG_CONSTRAINTS] = list(set(default_arg[ARG_CONSTRAINTS] + constraints))

	# Most common prepositionnal positions for each NP or PP arguments
	for complement_type, complement_info in default_subcat.items():
		for linked_arg in complement_info.keys():
			# if complement_type in [COMP_SUBJ, COMP_OBJ, COMP_IND_OBJ, COMP_PP]:
			# 	print(complement_type)
			# 	print(np.unique(complement_info[linked_arg][ARG_PREFIXES], return_counts=True))

			complement_info[linked_arg][ARG_PREFIXES] = list(set(complement_info[linked_arg][ARG_PREFIXES]))

	default_subcat[SUBCAT_OPTIONAL] = list(default_subcat.keys())
	default_entry[ENT_ORTH] = DEFAULT_ENTRY
	lexicon[DEFAULT_ENTRY] = default_entry

def simplify_lexicon(original_lexicon):
	"""
	Rearanges the given lexicon, and splitting it into two different lexicons:
	1. a verb lexicon that contains information only about verbs
	2. a nominalization lexicon that contians information only about nominalizations (including their plural forms)
	:param original_lexicon: a nomlex lexicon as a json format
	:return: the two differnt lexicons (verbs' lexicon and nominalizations' lexicon)
	"""
	global added_noms, removed_noms

	verbs_lexicon = {}
	noms_lexicon = {}

	tmp_lexicon = deepcopy(original_lexicon)

	# Simplify the lexicon keys by splitting entries and removing entries with errors
	removed_noms = remove_entries(tmp_lexicon)
	added_noms = split_entries(tmp_lexicon)
	print(f"Total removed entries: {len(removed_noms)}")
	print(f"Total added entries: {len(added_noms)}")
	print(f"Total entries after splitting and removing: {len(tmp_lexicon)}")

	# Rearanging any nominalization entry in the given lexicon
	for nominalization in tqdm(tmp_lexicon.keys(), "Rearranging the Lexicon", leave=False):
		curr_specs["word"] = tmp_lexicon[nominalization][ENT_ORTH]
		verb_entry, nom_entry, plural_entry = rearrange_entry(tmp_lexicon[nominalization])

		# Add the new entries to the appropriate
		new_nom = add_to_lexicon(noms_lexicon, nom_entry)
		new_verb = add_to_lexicon(verbs_lexicon, verb_entry)
		new_plural = add_to_lexicon(noms_lexicon, plural_entry)

		# Link the three entries (the ones that exists)
		if nom_entry is not None:
			nom_entry[ENT_VERB] = new_verb
			nom_entry[ENT_PLURAL] = new_plural

		if verb_entry is not None:
			verb_entry[ENT_NOM] = new_nom

		if plural_entry is not None:
			plural_entry[ENT_VERB] = new_verb
			plural_entry[ENT_SINGULAR] = new_nom

	print(f"Total verbs entries: {len(verbs_lexicon)}")
	print(f"Total noms entries: {len(noms_lexicon)}")

	if config.DEBUG:
		sanity_checks(verbs_lexicon, is_verb=True)
		sanity_checks(noms_lexicon, is_verb=False)
		get_summary()

	# Add an entry for a default verb\nom (words that don't appear in these lexicons)
	add_default_entry(verbs_lexicon)
	add_default_entry(noms_lexicon)

	return verbs_lexicon, noms_lexicon