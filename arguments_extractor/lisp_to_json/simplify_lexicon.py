from .simplify_lexicon_keys import *
from .simplify_entry import *

# For debug
added_noms = []
removed_noms = []
noms_with_missing_positions = []  # DET-POSS or N-N-MOD are missing
positions_by_type = defaultdict(set)
entries_by_type = defaultdict(set)
complements_per_subcat = defaultdict(set)



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
				raise Exception(f"Requires list isn't unique ({get_current_specs()}).")

			if len(set(optionals)) != len(optionals):
				raise Exception(f"Optionals list isn't unique ({get_current_specs()}).")

			# Check that the requires and the optionals lists aren't intersecting
			if set(difference_list(optionals, requires)) != set(optionals):
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

			for complement_type in all_complements:
				curr_specs["comp"] = complement_type
				complement_info = subcat[complement_type]
				for constraint in complement_info[ARG_CONSTRAINTS]:
					is_known(constraint, ["ARG_CONSTRAINT"], "ARG CONSTRAINTS")

				if (ARG_CONSTRAINT_DET_POSS_NO_OTHER_OBJ in complement_info[ARG_CONSTRAINTS] and POS_DET_POSS not in complement_info[ARG_CONSTANTS]) or\
				   (ARG_CONSTRAINT_N_N_MOD_NO_OTHER_OBJ in complement_info[ARG_CONSTRAINTS] and POS_N_N_MOD not in complement_info[ARG_CONSTANTS]):
					noms_with_missing_positions.append(word)

				positions_by_type["CONSTANTS"].update(complement_info[ARG_CONSTANTS])
				positions_by_type["PREFIXES"].update(complement_info[ARG_PREFIXES])
				positions_by_type["LINKED TO"].update(complement_info[ARG_LINKED].keys())

			curr_specs["comp"] = None
			more_argument_constraints = get_right_value(argument_constraints, subcat_type, {}, is_verb)

			for complement_type in more_argument_constraints.keys():
				curr_specs["comp"] = complement_type
				if complement_type not in subcat.keys():
					continue

				auto_controlled = []

				# Automatic constraints
				if complement_type.endswith("-P-OC"):
					auto_controlled = [COMP_PP]
				elif complement_type.endswith("-FOR-OC"):
					auto_controlled = [COMP_FOR_NP]
				elif complement_type.endswith("-POSSC"):
					auto_controlled = [COMP_POSS_ING_VC]
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
				if set(auto_controlled) != set(subcat[complement_type].get(ARG_CONTROLLED, [])):
					print(subcat[complement_type][ARG_CONTROLLED])
					print(auto_controlled)
					raise Exception(f"Manual controlled constraints do not agree with the automatic ones ({get_current_specs()}).")

				if subcat[complement_type][ARG_LINKED] == {} and subcat[complement_type][ARG_CONSTANTS] == subcat[complement_type][ARG_PREFIXES] == []:
					print(word, subcat_type, complement_type)
					raise Exception(f"There is a complement without any position ({get_current_specs()}).")

			curr_specs["comp"] = None

def get_summary():
	print("\nValues that were found during the creation of the new representation:")
	print("----------------------")

	print(f"Possible nom roles for PVAL: {nom_roles_for_pval}")
	print(f"Required arguments that are sometimes missing: {list(zip(*np.unique(missing_required, return_counts=True)))}")
	print(f"Words with missing DET-POSS/N-N-MOD (based on NO-OTHER-OBJ feature): {noms_with_missing_positions}")
	print(f"Complements that don't specify specific postag: {args_without_pos}")

	for constant_type in list(set(list(unknown_values_dict.keys()) + list(known_values_dict.keys()))):
		print(f"\n{constant_type}:")
		print("----------------------")
		print("RELEVANT:", known_values_dict.get(constant_type, {}))
		print("IGNORED:", unknown_values_dict.get(constant_type, {}))

	print("\nPossible positions by type:")
	print("----------------------")
	for positions_type, positions_list in positions_by_type.items():
		print(f"{positions_type}: {positions_list}")

	print("\nRelevant entries by type:")
	print("----------------------")
	for entries_type, entries_list in entries_by_type.items():
		print(f"{entries_type.__name__}: {entries_list}")



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

	if DEBUG:
		sanity_checks(verbs_lexicon, is_verb=True)
		sanity_checks(noms_lexicon, is_verb=False)
		get_summary()

	return verbs_lexicon, noms_lexicon