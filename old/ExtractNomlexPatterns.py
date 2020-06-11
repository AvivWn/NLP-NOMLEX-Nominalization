import itertools
from collections import defaultdict

import DictsAndTables
from DictsAndTables import subentries_table, special_subcats_dict, \
						   get_clean_nom



########################################## Extracting NOMLEX Patterns ###########################################

def update_options(options, info, subentry=None):
	"""
	Updates the options in the right way
	:param options: the current possible options (list)
	:param info: where to get alternatives (for PP)
	:param subentry: helps to know where to get alternatives (for PP), optional
	:return: the updated possible options
	"""

	if "NOM" in options:
		return options

	if subentry == "SUBJECT" or not subentry:
		options.append("PP-BY")

		if "NOT-PP-BY" in options:
			options.remove("PP-BY")
			options.remove("NOT-PP-BY")

	elif subentry == "IND-OBJ":
		if "IND-OBJ-OTHER" in options:
			options.remove("IND-OBJ-OTHER")

			other_info = info[subentry]["IND-OBJ-OTHER"]
			options += ["PP-" + s.upper() for s in list(other_info.values())[0]]

	if "PP" in options:
		options.remove("PP")

		if subentry:
			PP_info = info[subentry]["PP"]
		else:
			PP_info = info["PP"]

		if len(PP_info.get("PVAL", [])) > 0:
			options += ["PP-" + s.upper() for s in list(PP_info.get("PVAL", []))]

	options = list(set(options))

	return options

def get_options(a_list, order_required):
	"""
	Returns all the different possibilities of the arguments in the given list
	:param a_list: a list of arguments (each argument of several possible values = a list)
	:param order_required: is the order (of pvals- pval and pval2) improtant
	:return: all the possible tuples, also the illegal ones
	"""

	all_tuples = list(itertools.product(*a_list))
	subentries_types = [i[0] for i in subentries_table]

	if a_list[subentries_types.index("pval")] != ["NONE"] and \
	   		a_list[subentries_types.index("pval2")] != ["NONE"] and \
			not order_required:
		temp = a_list[subentries_types.index("pval")]
		a_list[subentries_types.index("pval")] = a_list[subentries_types.index("pval2")]
		a_list[subentries_types.index("pval2")] = temp
		all_tuples += list(itertools.product(*a_list))

	return all_tuples

def get_subentries(subcat_info, subcat, default_subjects):
	"""
	Returns a list of the possible types of each subentry in the given subcat
	The subentries has a specific order in the resulted list
	If the subentry doesn't appear in the given subcat, its possible types are ['NONE']
	:param subcat_info: a dictionary of the subcat's information
	:param subcat: the subcat's name
	:param default_subjects: the default list of subject types
	:return: a list of lists (its order defined by the get_subentries_order() function)
	"""

	subentries_types_table = subentries_table
	subentries_types = [i[0] for i in subentries_table]

	subentries_options = []

	# Moving over all the possible subentries
	for i in range(len(subentries_types_table)):
		subentry_options = []
		default_subentry_value = []

		# Are there any special cases? checking it with the suitable dictionary
		if subcat in special_subcats_dict.keys():
			# Initiating the relevant subentries with the suitable default value
			for special_subentry, default_value in special_subcats_dict[subcat][1]:
				if special_subentry == subentries_types_table[i][0]:
					default_subentry_value = default_value

		# Getting the subentry options from the given subcat info
		for how_to_find_subentry, limited_subcats, exception_subcats in subentries_types_table[i][2]:
			if (limited_subcats == [] or subcat in limited_subcats) and subcat not in exception_subcats:
				to_subentry = subcat_info
				for x in how_to_find_subentry:
					to_subentry = to_subentry.get(x, {})

				# Adding "PP-" before the preposition values (for ind-object only)
				if how_to_find_subentry == ["PVAL1"] and to_subentry not in ["*NONE*", "NONE"] and subentries_types_table[i][0] == "ind-object":
					for j in range(len(to_subentry)):
						to_subentry[j] = "PP-" + to_subentry[j].upper()

				if type(to_subentry) == str:
					to_subentry = [to_subentry]

				if subentry_options in ["NONE", "*NONE*"]:
					subentry_options = to_subentry
				else:
					subentry_options += to_subentry

				# Dealing with the subentry results
				if subentry_options == ["NONE"]:
					subentry_options = "NONE"
				elif subentry_options == ["*NONE*"]:
					subentry_options = "*NONE*"
				else:
					if how_to_find_subentry in [["OBJECT"], ["IND-OBJ"]]:
						subentry_options = update_options(subentry_options, subcat_info, how_to_find_subentry[0])
					elif how_to_find_subentry == ["SUBJECT"]:
						if subentry_options == []:
							subentry_options = default_subjects
						else:
							subentry_options = update_options(subentry_options, subcat_info, how_to_find_subentry[0])

		if subentry_options == []:
			subentry_options = default_subentry_value

			if subentry_options == []:
				subentry_options = ["NONE"]
		elif subentry_options == "*NONE*":
			subentry_options = ['NONE']

		subentries_options.append(subentry_options)

	# Some manual updates

	pval_subentry = subentries_options[subentries_types.index("pval-nom")]
	if pval_subentry != ["NONE"]:
		subentries_options[subentries_types.index("pval")] = ["pval-nom"]

	pval_subentry = subentries_options[subentries_types.index("pval1-nom")]
	if pval_subentry != ["NONE"]:
		subentries_options[subentries_types.index("ind-object")] = ["pval1-nom"]

	pval_subentry = subentries_options[subentries_types.index("pval2-nom")]
	if pval_subentry != ["NONE"]:
		subentries_options[subentries_types.index("pval2")] = ["pval2-nom"]

	wh_subentry_options = subentries_options[subentries_types.index("wh")]
	if wh_subentry_options != ["NONE"]:
		wh_subentry = []
		for wh_subentry_option in wh_subentry_options:
			if subcat in ["NOM-WH-S", "NOM-NP-WH-S", "NOM-PP-WH-S"]:
				wh_subentry.append(wh_subentry_option + " whether")
				wh_subentry.append(wh_subentry_option + " what")
			elif subcat == "NOM-HOW-S":
				wh_subentry.append(wh_subentry_option + " how")
		subentries_options[subentries_types.index("wh")] = wh_subentry.copy()

	if subcat == "NOM-PP-HOW-TO-INF":
		subentries_options[subentries_types.index("pval1")].append("NONE")

	#if DictsAndTables.should_print and DictsAndTables.should_print_to_screen: print(subentries_options)

	return subentries_options

def clean_patterns(patterns, entry, subcat_info, subcat):
	"""
	Cleans the illegal patterns from the patterns list
	:param patterns: the extracted patterns in a list
	:param entry: the specific nominalization entry
	:param subcat_info: the subcat information
	:param subcat: the specific sub-categorization
	:return: The cleaned patterns list
	"""

	# Maintaining individuality of patterns
	patterns = list(set(patterns))

	# Removing patterns that we know that cannot appear according to the NOMLEX manual
	for pattern in patterns:
		# Too short pattern
		if len(list(set(pattern))) == 1 and subcat != "NOM-INTRANS" and subcat != "NOM-INTRANS-RECIP":
			patterns.remove(pattern)

		# More than one possessive determiner in the pattern
		elif len([x for x in list(pattern) if x == 'DET-POSS']) >= 2:
			patterns.remove(pattern)

		# Both argument as PP and pval written after 'of' preposition
		elif 'PP-OF' in list(pattern) and 'of' in list(pattern):
			patterns.remove(pattern)

	# Removing patterns according to the "NOT" tag
	not_subentry = subcat_info.get("NOT", {})
	for and_entry, _ in not_subentry.items():
		not_patterns = get_nom_subcat_patterns(entry, not_subentry, and_entry)

		for not_pattern in not_patterns:
			if not_pattern in patterns:
				patterns.remove(not_pattern)

	return patterns

def get_nom_subcat_patterns(entry, main_subentry, subcat):
	"""
	Creates and returns the patterns for a suitable nominalization entry and sub-categorization
	:param entry: the specific nominalization entry
	:param subcat: the specific sub-categorization
	:param main_subentry: the main entry to search for the given subcat.
						  If the main entry is None, than the entry is the default "VERB-SUBC" entry.
	:return: a list of patterns, each pattern is a dictionary (can be also tuple sometimes)
	"""

	# Getting the default subject roles
	verb_subj_info = entry.get("VERB-SUBJ", {"NONE": {}})

	if verb_subj_info == "NONE":
		verb_subj_info = {"NONE": {}}

	default_subjects = update_options(list(verb_subj_info.keys()), verb_subj_info)

	patterns = []

	# Getting the required list
	required_list = list(entry.get("REQUIRED", {}).keys())

	# Trying to get object, subjects, and indirect-objects and other sub-entries (in the given subcat)
	if main_subentry:
		subcat_info = main_subentry.get(subcat, "NONE")
	else:
		subcat_info = entry.get("VERB-SUBC", {}).get(subcat, "NONE")

	# Continue only if the subcat entry exists
	if subcat_info == "NONE":
		return []

	# Finding the subentries possible types
	subentries = get_subentries(subcat_info, subcat, default_subjects)
	subentries_types = [i[0] for i in subentries_table]

	# Is the nominalization itself has a role in the sentence
	# Here we ignore cases that NOM-TYPE is SUBJECT + OBJECT or SUBJECT\OBJECT + VERB-NOM
	if list(entry["NOM-TYPE"].keys()) == ["SUBJECT"]:
		subentries[subentries_types.index("subject")] = ["NOM"]
	elif list(entry["NOM-TYPE"].keys()) == ["OBJECT"]:
		subentries[subentries_types.index("object")] = ["NOM"]

	# Getting required values from the special subcats dictionary
	special_subcats_cases = special_subcats_dict

	if subcat in special_subcats_cases:
		required_list += special_subcats_cases[subcat][0]

	order_required = "order" in required_list

	subjects = subentries[subentries_types.index("subject")]
	objects = subentries[subentries_types.index("object")]

	# Creating some patterns for the suitable case
	if subjects != "NONE" and objects != "NONE":
		# Without the subject
		if "SUBJECT" not in required_list and subjects != ["NOM"]:
			temp_subentries = subentries.copy()
			temp_subentries[subentries_types.index("subject")] = ["NONE"]
			patterns += get_options(temp_subentries, order_required)

		# Without the object
		if "OBJECT" not in required_list and objects != ["NOM"]:
			temp_subentries = subentries.copy()
			temp_subentries[subentries_types.index("object")] = ["NONE"]
			patterns += get_options(temp_subentries, order_required)

		# With the subject and the object
		if "SUBJECT" not in required_list and subjects != ["NOM"] and "OBJECT" not in required_list and objects != ["NOM"]:
			temp_subentries = subentries.copy()
			temp_subentries[subentries_types.index("object")] = ["NONE"]
			temp_subentries[subentries_types.index("subject")] = ["NONE"]
			patterns += get_options(temp_subentries, order_required)

		patterns += get_options(subentries, order_required)

	# Only the object
	elif objects != "NONE" and "OBJECT" not in required_list:
		subentries[subentries_types.index("subject")] = ["NONE"]
		patterns += get_options(subentries, order_required)

	# Only the subject
	elif subjects != "NONE" and "SUBJECT" not in required_list:
		subentries[subentries_types.index("object")] = ["NONE"]
		patterns += get_options(subentries, order_required)

	# Deleting illegal patterns
	patterns = clean_patterns(patterns, entry, subcat_info, subcat)

	# Translating each pattern to dictionary (from list)
	dicts_patterns = []
	for pattern in patterns:
		pattern = list(pattern)
		dict_pattern = defaultdict(str)
		dict_pattern["subcat"] = subcat
		dict_pattern["verb"] = entry["VERB"]

		for i in range(len(subentries_types)):
			if pattern[i] != "NONE":
				dict_pattern[subentries_types[i]] = pattern[i]

		dicts_patterns.append(dict_pattern)

	if main_subentry:
		return patterns
	else:
		return dicts_patterns

def get_nom_patterns(entry, subcat=None):
	"""
	Returns the possible object and subject pairs for the given entry
	:param entry: a dictionary info of a specific nominalization
	:param subcat: a sub-categorization type, optional argument.
		   If subcat is None, than the extraction won't be specific for a given subcat.
	:return: a list of all possible pairs for a specific nominalization entry (list of tuples)
	"""

	patterns = []

	if subcat:
		patterns += get_nom_subcat_patterns(entry, None, subcat)
	else:
		for subcat in entry.get("VERB-SUBC", {}).keys():
			patterns += get_nom_subcat_patterns(entry, None, subcat)

	return patterns

def extract_nom_patterns(nomlex_entries, subcat=None):
	"""
	Extracts all the nominalization patterns from the given nomlex entries
	:param nomlex_entries: the json formatted data to extract from (entries)
						   This argument can be the whole NOMLEX lexicon, or only few entries from their
	:param subcat: a sub-categorization type, optional argument.
		   		   If subcat is None, than the extraction won't be specific for a given subcat.
	:return: the nominalization patterns that can be found in the given entries (dictionary {nom: (clean_nom, [pattern])})
	"""

	patterns_dict = {}

	for nominalization, entry in nomlex_entries.items():
		patterns = get_nom_patterns(entry, subcat=subcat)
		patterns_dict.update({nominalization: (get_clean_nom(nominalization), patterns)})

	return patterns_dict



######################################### Aggregating NOMLEX Patterns ###########################################

def aggregate_patterns(patterns_dict):
	"""
	Aggregates all the patterns in the given dictionary into a single list of patterns
	:param patterns_dict: a dictionary of patterns, a list of pattern for each nominalization
	:return: a list of all the aggreated patterns, each pattern appear twice.
			 The patterns don't include information about "verb" or "subcat"
	"""

	unique_patterns = []

	possible_subcats = [row[0] for row in DictsAndTables.comlex_table]

	for _, possible_arguments in patterns_dict.values():
		for args_dict in possible_arguments:
			if args_dict["subcat"] in possible_subcats:
				del args_dict["verb"]
				del args_dict["subcat"]

				if args_dict != {}:
					if args_dict not in unique_patterns:
						unique_patterns.append(args_dict)

	return unique_patterns