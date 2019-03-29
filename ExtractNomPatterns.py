import json
import itertools
from allennlp.predictors.constituency_parser import ConstituencyParserPredictor
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from collections import defaultdict
import spacy
import inflect

inflect_engine = inflect.engine()
nlp = spacy.load('en_core_web_sm')

det = "the"
should_print = True

############################################# Dictionaries and Tables ############################################

def get_subentries_table():
	# (subentry, ud_links, [(how_to_find, limited_subcats, exception_subcats)])
	# limited_subcats = [] means not limited
	# exception_subcats = [] means with no exceptions
	return [
		("subject", 		{"DET-POSS":["poss"], "N-N-MOD":["compound"], "PP-":["prep_", "pobj"]},	[(["SUBJECT"], [], [])]),
		("ind-object", 		{"DET-POSS":["poss"], "N-N-MOD":["compound"], "PP-":["prep_", "pobj"]},	[(["PVAL1"], [], ["NOM-PP-FOR-TO-INF", "NOM-PP-TO-INF-RECIP"]), ("IND-OBJ", [], [])]),
		("object", 			{"DET-POSS":["poss"], "N-N-MOD":["compound"], "PP-":["prep_", "pobj"]},	[(["OBJECT"], [], ["NOM-NP-ING", "NOM-NP-ING-SC", "NOM-NP-ING-OC"])]),
		("pval", 			["prep_", ["pobj"]],													[(["PVAL"], [], ["NOM-P-NP-ING", "NOM-NP-P-NP-ING"])]),
		("pval1", 			["prep_", ["pobj"]],													[(["PVAL1"], ["NOM-PP-FOR-TO-INF", "NOM-PP-TO-INF-RECIP"], [])]),
		("pval2", 			["prep_", ["pobj"]],													[(["PVAL2"], [], [])]),
		("pval-nom", 		[],																		[(["PVAL-NOM"], [], [])]),
		("pval1-nom", 		[],																		[(["PVAL1-NOM"], [], [])]),
		("pval2-nom", 		[],																		[(["PVAL2-NOM"], [], [])]),
		("pval-ing", 		["prep_", ["pcomp__ing"]],												[(["NOM-SUBC", "P-ING", "PVAL"], [], ["NOM-ING-SC"])]), # P-ING
		("pval-comp-ing", 	["prep_", ["pobj"], ["pcomp__ing"]],									[(["PVAL"], ["NOM-P-NP-ING", "NOM-NP-P-NP-ING"], [])]), # P-NP-ING
		("pval-to-inf", 	["advcl", ["mark_"], ["aux_to"]],										[]), # P-TO-INF
		("comp-ing", 		["prep_", "pobj__ing"],													[(["OBJECT"], ["NOM-NP-ING", "NOM-NP-ING-SC", "NOM-NP-ING-OC"], [])]), # NP-ING
		("ing", 			["prep_", "pcomp__ing"],												[(["NOM-SUBC", "P-ING", "PVAL"], ["NOM-ING-SC"], [])]), # just ING
		("adverb", 			{"ADJP": ["amod"], "ADVP": ["advmod"]},									[(["NOM-SUBC"], ["NOM-ADVP-PP", "NOM-NP-ADVP", "NOM-ADVP"], [])]),
		("sbar", 			["acl", ["mark_that"]], 												[]),
		("adjective", 		["prep_", "amod"], 														[]),
		("to-inf", 			["acl", ["aux_to"]],													[]), # TO-INF
	]

def get_special_subcats_dict():
	# subcat: (required_list, list of (subentry, default_value) pairs)
	# About the required_list- should contain SUBJECT, OBJECT and order.
	# However, all the other subentries always required (according to the subcat structure).
	return {
		"NOM-NP":					(["OBJECT"],			[]),

		"NOM-ADVP":					([],					[("adverb", ["ADVP"])]),
		"NOM-ADVP-PP":				([],					[("adverb", ["ADVP"])]),
		"NOM-NP-ADVP":				(["OBJECT"],			[("adverb", ["ADVP"])]),

		"NOM-NP-TO-NP":				(["OBJECT"],			[("ind-object", ["PP-TO"])]),
		"NOM-NP-FOR-NP":			(["OBJECT"],			[("ind-object", ["PP-FOR"])]),

		"NOM-NP-AS-NP-SC":			(["OBJECT", "SUBJECT"],	[("pval", ["as"])]),
		"NOM-NP-AS-NP":				([],					[("pval", ["as"])]),
		"NOM-AS-NP":				([],					[("pval", ["as"])]),
		"NOM-NP-PP-AS-NP":			(["OBJECT", "order"],	[("pval2", ["as"])]),
		"NOM-NP-PP-PP":				(["OBJECT"],			[]),
		"NOM-NP-PP":				(["OBJECT"],			[]),

		"NOM-NP-AS-ADJP":			(["OBJECT"],			[("adjective", ["as"])]),

		"NOM-S":					([],					[("sbar", ["NOT NONE"])]),
		"NOM-THAT-S":				([],					[("sbar", ["NOT NONE"])]),
		"NOM-NP-S":					(["OBJECT"],			[("sbar", ["NOT NONE"])]),
		"NOM-PP-THAT-S":			([],					[("sbar", ["NOT NONE"])]),

		"NOM-NP-AS-ING":			(["OBJECT"],			[("pval-ing", ["as"])]),
		"NOM-NP-P-NP-ING":			(["OBJECT"],			[]),
		"NOM-NP-P-ING":				(["OBJECT"],			[]),
		"NOM-NP-P-ING-OC":			(["OBJECT"],			[]),
		"NOM-NP-P-ING-SC":			(["OBJECT"],			[]),

		"NOM-PP-FOR-TO-INF":		([],					[("pval-to-inf", ["for"])]),
		"NOM-FOR-TO-INF":			([],					[("pval-to-inf", ["for"])]),
		"NOM-PP-TO-INF-RECIP":		([],					[("to-inf", ["to"])]),
		"NOM-P-NP-TO-INF":			([],					[("to-inf", ["to"])]),
		"NOM-P-NP-TO-INF-OC":		([],					[("to-inf", ["to"])]),
		"NOM-P-NP-TO-INF-VC":		([],					[("to-inf", ["to"])]),
		"NOM-NP-TO-INF-SC":			(["OBJECT"],			[("to-inf", ["to"])]),
		"NOM-NP-TO-INF-OC":			(["OBJECT"],			[("to-inf", ["to"])]),
		"NOM-NP-TO-INF-VC":			(["OBJECT"],			[("to-inf", ["to"])])
	}

def get_comlex_table():
	# subcat, structure, suitable_pattern_entities
	# Be aware that the order matter, because the program try each line in that order and we want to find the most specific case
	comlex_table = [
		 ("NOM-PP-FOR-TO-INF",			["PP", ["IN_for", ["NP", ["TO_to", ["VB"]]]]],	["pval1", "pval-to-inf"]),						# NEW
		 ("NOM-FOR-TO-INF",				[["IN_for", ["NP", ["TO_to", ["VB"]]]]],		["pval-to-inf"]),								# NEW
		 ("NOM-PP-TO-INF-RECIP",		["PP", [["TO_to", ["VB"]]]],					["pval1", "to-inf"]),							# NEW
		 ("NOM-P-NP-TO-INF",			[["IN", "NP"], [["TO_to", ["VB"]]]],			["pval", "to-inf"]),							# NEW
		 ("NOM-P-NP-TO-INF-OC",			[["IN", "NP"], [["TO_to", ["VB"]]]],			["pval", "to-inf"]),							# NEW
		 ("NOM-P-NP-TO-INF-VC",			[["IN", "NP"], [["TO_to", ["VB"]]]],			["pval", "to-inf"]),							# NEW
		 ("NOM-NP-TO-INF-VC",			["NP", [["TO_to", ["VB"]]]],					["object", "to-inf"]),							# NEW
		 ("NOM-NP-TO-INF-SC",			["NP", [["TO_to", ["VB"]]]],					["object", "to-inf"]),							# NEW
		 ("NOM-NP-TO-INF-OC",			["NP", [["TO_to", ["VB"]]]],					["object", "to-inf"]),							# NEW
		 ("NOM-TO-INF-SC",				[[["TO_to", ["VB"]]]],							["to-inf"]),									# OK

		 # ING- gerunds
		 ("NOM-NP-P-NP-ING",			["NP", ["IN", ["NP", ["VBG"]]]],				["object", "pval-comp-ing"]),					# OK
		 ("NOM-NP-P-NP-ING",			["NP", ["IN", ["NP", [["VBG"]]]]],				["object", "pval-comp-ing"]),					# OK
		 ("NOM-P-NP-ING",				[["IN", ["NP", ["VBG"]]]],						["pval-comp-ing"]),								# OK
		 ("NOM-P-NP-ING",				[["IN", ["NP", [["VBG"]]]]],					["pval-comp-ing"]),								# OK
		 ("NOM-NP-AS-ING",				["NP", ["IN_as", ["VBG"]]],						["object", "pval-ing"]),						# OK
		 ("NOM-NP-AS-ING",				["NP", ["IN_as", [["VBG"]]]],					["object", "pval-ing"]),						# OK
		 ("NOM-NP-P-ING",				["NP", ["IN", ["VBG"]]],						["object", "pval-ing"]),						# OK
		 ("NOM-NP-P-ING",				["NP", ["IN", [["VBG"]]]],						["object", "pval-ing"]),						# OK
		 ("NOM-NP-P-ING-OC",			["NP", ["IN", ["VBG"]]],						["object", "pval-ing"]),						# OK
		 ("NOM-NP-P-ING-OC",			["NP", ["IN", [["VBG"]]]],						["object", "pval-ing"]),						# OK
		 ("NOM-NP-P-ING-SC",			["NP", ["IN", ["VBG"]]],						["object", "pval-ing"]),						# OK
		 ("NOM-NP-P-ING-SC",			["NP", ["IN", [["VBG"]]]],						["object", "pval-ing"]),						# OK
		 ("NOM-P-ING-SC",				[["IN", ["VBG"]]],								["pval-ing"]),									# OK
		 ("NOM-P-ING-SC",				[["IN", [["VBG"]]]],							["pval-ing"]),									# OK
		 ("NOM-NP-ING",					[["NP", ["VBG"]]],								["comp-ing"]),									# OK
		 ("NOM-NP-ING",					[["NP", [["VBG"]]]],							["comp-ing"]),									# OK
		 ("NOM-NP-ING-OC",				[["NP", ["VBG"]]],								["comp-ing"]),									# OK
		 ("NOM-NP-ING-OC",				[["NP", [["VBG"]]]],							["comp-ing"]),									# OK
		 ("NOM-NP-ING-SC",				["NP", ["VBG"]],								"comp-ing"),									# OK
		 ("NOM-NP-ING-SC",				["NP", [["VBG"]]],								"comp-ing"),									# OK
		 ("NOM-ING-SC",					[["VBG"]],										["ing"]),										# OK
		 ("NOM-ING-SC",					[[["VBG"]]],									["ing"]),										# OK

		 # SBAR
		 ("NOM-PP-THAT-S",				[["IN", "NP"], ["IN_that", "S"]],				[[None, "ind-object"], "sbar"]),				# OK
		 ("NOM-NP-S",					["NP", ["S"]],									["object", "sbar"]),							# OK
		 ("NOM-NP-S",					["NP", ["IN_that", "S"]],						["object", "sbar"]),							# OK
		 ("NOM-THAT-S",					[["IN_that", "S"]],								["sbar"]),										# OK
		 ("NOM-S",						[["S"]],										["sbar"]),										# OK
		 ("NOM-S",						[["IN_that", "S"]],								["sbar"]),										# OK

		 # Double pvals
		 ("NOM-NP-PP-AS-NP",			["NP", ["IN", "NP"], ["IN_as", "NP"]],			["object", [None, "ind-object"], "pval2"]),
		 ("NOM-NP-PP-AS-NP",			[["NP", ["IN", "NP"]], ["IN_as", "NP"]],		[["object", [None, "ind-object"]], "pval2"]),
		 ("NOM-NP-PP-PP",				["NP", "PP", "PP"],								["object", "pval", "pval2"]),					# OK
		 ("NOM-NP-PP-PP",				[["NP", "PP"], "PP"],							[["object", "pval"], "pval2"]),					# OK
		 ("NOM-PP-PP",					["PP", "PP"],									["pval", "pval2"]),								# OK

		 # Both object and indirect-object
		 ("NOM-NP-TO-NP",				["NP", ["IN_to", "NP"]],						["object", [None, "ind-object"]]),				# OK
		 ("NOM-NP-TO-NP",				[["IN_to", "NP"], "NP"],						[[None, "ind-object"], "object"]),				# OK
		 ("NOM-NP-TO-NP",				["NP", "NP"],									["ind-object", "object"]),						# OK
		 ("NOM-NP-FOR-NP",				["NP", ["IN_for", "NP"]],						["object", [None, "ind-object"]]),				# OK
		 ("NOM-NP-FOR-NP",				[["IN_for", "NP"], "NP"],						[[None, "ind-object"], "object"]),				# OK
		 ("NOM-NP-FOR-NP",				["NP", "NP"],									["ind-object", "object"]),						# OK

		 # Adjective
		 ("NOM-NP-AS-ADJP",				["NP", ["RB_as", "JJ"]],						["object", "adjective"]),						# OK

		 # Single pval
		 ("NOM-NP-AS-NP-SC",			["NP", ["IN_as", "NP"]],						["object", "pval"]),							# OK
		 ("NOM-NP-AS-NP",				["NP", ["IN_as", "NP"]],						["object", "pval"]),							# OK
		 ("NOM-AS-NP",					[["IN_as", "NP"]],								["pval"]),										# OK
		 ("NOM-NP-PP",					["NP", "PP"],									["object", "pval"]),							# OK

		 # Double objects
		 ("NOM-NP-NP",					["NP", "NP"],									["ind-object", "object"]),						# OK

		 # Adverb
		 ("NOM-ADVP-PP",				["ADVP", "PP"],									["adverb", "pval"]),							# OK
		 ("NOM-NP-ADVP",				["NP", "ADVP"],									["object", "adverb"]),							# OK
		 ("NOM-ADVP",					["ADVP"],										["adverb"]),									# OK

		 # Basic
		 ("NOM-PP",						["PP"],											["pval"]),										# OK
		 ("NOM-NP",						["NP"],											["object"])										# OK
	]

	return comlex_table


def get_pronoun_dict():
	pronoun_dict = {"he":["his", "him"], "she":["her", "her"], "it":["its", "its"], "they":["their", "them"], "we":["our", "us"], "i":["my", "me"]}

	return pronoun_dict




############################################### Extracting Patterns ##############################################

def update_option(options, info, role=None):
	"""
	Updates the options in the right way
	:param options: the current possible options (list)
	:param info: where to get alternatives (for PP)
	:param role: helps to know where to get alternatives (for PP)
	:return: the updated possible options
	"""

	if "NOM" in options:
		return options

	if role == "SUBJECT" or not role:
		options.append("PP-BY")

		if "NOT-PP-BY" in options:
			options.remove("PP-BY")
			options.remove("NOT-PP-BY")

	elif role == "IND-OBJ":
		if "IND-OBJ-OTHER" in options:
			options.remove("IND-OBJ-OTHER")

			other_info = info[role]["IND-OBJ-OTHER"]
			options += ["PP-" + s.upper() for s in list(other_info.values())[0]]

	if "PP" in options:
		options.remove("PP")

		if role:
			PP_info = info[role]["PP"]
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
	:param order_required: is the order (of pvals) is improtant
	:return: all the possible tuples, also the illegal ones
	"""

	all_tuples = list(itertools.product(*a_list))
	subentries_types = [i[0] for i in get_subentries_table()]

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

	subentries_types_table = get_subentries_table()
	subentries_types = [i[0] for i in get_subentries_table()]

	special_cases = get_special_subcats_dict()

	subentries = []

	# Moving over all the possible subentries
	for i in range(len(subentries_types_table)):
		subentry = []
		default_subentry_value = []

		# Are there any special cases? checking it with the suitable dictionary
		if subcat in special_cases.keys():
			# Initiating the relevant subentries with the suitable default value
			for special_subentry, default_value in special_cases[subcat][1]:
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

				if subentry == "NONE":
					subentry = to_subentry
				else:
					subentry += to_subentry

				# Dealing with the subentry results
				if subentry == ["NONE"] or subentry == ["*NONE*"]:
					subentry = "NONE"
				else:
					if how_to_find_subentry in [["OBJECT"], ["IND-OBJ"]]:
						subentry = update_option(subentry, subcat_info, how_to_find_subentry[0])
					elif how_to_find_subentry == ["SUBJECT"]:
						if subentry == []:
							subentry = default_subjects
						else:
							subentry = update_option(subentry, subcat_info, how_to_find_subentry[0])

		if subentry == []:
			subentry = default_subentry_value

			if subentry == []:
				subentry = ["NONE"]

		subentries.append(subentry)

	pval_subentry = subentries[subentries_types.index("pval-nom")]
	if pval_subentry != ["NONE"]:
		subentries[subentries_types.index("pval")] = ["pval-nom"]

	pval_subentry = subentries[subentries_types.index("pval1-nom")]
	if pval_subentry != ["NONE"]:
		subentries[subentries_types.index("ind-object")] = ["pval1-nom"]

	pval_subentry = subentries[subentries_types.index("pval2-nom")]
	if pval_subentry != ["NONE"]:
		subentries[subentries_types.index("pval2")] = ["pval2-nom"]

	#if should_print: print(subentries)

	return subentries

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

	default_subjects = update_option(list(verb_subj_info.keys()), verb_subj_info)

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

	subentries_types = [i[0] for i in get_subentries_table()]

	# TODO: dealing with nominalization like prayer
	# Is the nominalization itself has a role in the sentence
	if "SUBJECT" in entry["NOM-TYPE"].keys():
		subentries[subentries_types.index("subject")] = ["NOM"]
	elif "OBJECT" in entry["NOM-TYPE"].keys():
		subentries[subentries_types.index("object")] = ["NOM"]

	# Getting required values from the special subcats dictionary
	special_subcats_cases = get_special_subcats_dict()

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

def extract_nom_patterns(entries, subcat=None):
	"""
	Extracts all the nominalization patterns from the given nomlex entries
	:param entries: the json formatted data to extract from (entries)
	:param subcat: a sub-categorization type, optional argument.
		   If subcat is None, than the extraction won't be specific for a given subcat.
	:return: the nominalization patterns that can be found in the given entries
	"""

	patterns_list = []
	patterns_dict = {}

	for nominalization, entry in entries.items():
		patterns = get_nom_patterns(entry, subcat=subcat)
		patterns_dict.update({nominalization: patterns})
		patterns_list += patterns

	return patterns_dict




################################################ Verbal to Nominal ###############################################

def get_nomlex_entries(entries, verb):
	"""
	Returns the relevant nominalization entries for a specific verb
	:param entries: a dictionary of all the entries in NOMLEX lexicon
	:param verb: the base verb
	:return: a dictionary that contain only the relevant entries for the given verb
	"""

	relevant_entries = {}

	for nom, entry in entries.items():
		if "VERB" in entry.keys() and entry["VERB"] == verb:
			relevant_entries.update({nom: entry})

	return relevant_entries

def process_phrases_tree(sent_phrases_tree, index):
	"""
	A recursive function that processes a phrases tree as string and returns the suitable dictionary representation of the same tree
	:param sent_phrases_tree: a phrases tree in string format
	:param index: the current index in the sentence
	:return: a dictionary that represents the given phrases tree
	"""

	sub_phrases_trees = []

	while sent_phrases_tree[index] != ")":
		if sent_phrases_tree[index] == "(":
			index, sub_phrases_tree = process_phrases_tree(sent_phrases_tree, index + 1)
			sub_phrases_trees.append(sub_phrases_tree)
		else:
			sub_phrases_trees.append(sent_phrases_tree[index])
			index += 1

	if len(sub_phrases_trees) == 2:
		new_phrase_tree = {sub_phrases_trees[0]: [sub_phrases_trees[1]]}
	else:
		new_phrase_tree = {sub_phrases_trees[0]: sub_phrases_trees[1:]}

	return index + 1, new_phrase_tree

def get_phrase(phrases_tree):
	"""
	Returns the phrase that the given phrase tree represents
	:param phrases_tree: a given phrases tree (dictionary)
	:return: a string value
	"""

	# String- a single word of a phrase (a leaf)
	if type(phrases_tree) == str:
		return phrases_tree

	str_phrase = ""

	# List- the subtrees of the phrases tree
	if type(phrases_tree) == list:
		for sub_phrases_tree in phrases_tree:
			if str_phrase != "":
				str_phrase += " "

			str_phrase += get_phrase(sub_phrases_tree)

		return str_phrase

	# Dictionary- a subtree of the phrases tree
	for _, sub_phrases_tree in phrases_tree.items():
		if type(sub_phrases_tree) == str:
			if str_phrase != "":
				str_phrase += " "

			str_phrase += sub_phrases_tree
		else:
			for sub_sub_phrases_tree in sub_phrases_tree:
				sub_sub_phrase = get_phrase(sub_sub_phrases_tree)

				if str_phrase != "":
					str_phrase += " "

				str_phrase += sub_sub_phrase

	return str_phrase

def search_phrase(phrases_tree, searched_tag):
	"""
	A recursive function that searches for a specific phrase in the given tree
	:param phrases_tree: the given tree that is being searched
	:param searched_tag: the wanted tag
	:return: a list of all the trees with the wanted phrase tag as root
	"""

	if type(phrases_tree) == str:
		return []

	wanted_phrases = []

	for phrase_tag, sub_phrase_tree in phrases_tree.items():
		if phrase_tag == searched_tag:
			wanted_phrases.append({phrase_tag: sub_phrase_tree})
		else:
			for sub_sub_phrase_tree in sub_phrase_tree:
				sub_wanted_phrases = search_phrase(sub_sub_phrase_tree, searched_tag)

				if sub_wanted_phrases != []:
					wanted_phrases += sub_wanted_phrases

	return wanted_phrases

def get_sub_phrases(phrases_tree, phrases_tags, arguments_types = None):
	"""
	This function tries to find the phrases tags in the given list as sub-phrases of the given phrases tree
	It also may match the founded phrases to the given argument types according to their order (only if
	:param phrases_tree: a dictionary that represents the phrases tree to search in
	:param phrases_tags: a phrases tags list, we are looking to find
	:param arguments_types: a list of types of arguments to match with the wanted phrases
	:return: the list of sub-phrases that were found (if all the sub-phrases tags were found) or []
			 In addition, the matching arguments dictionary is also returned (if the arguments types aren't given, None is returned)
	"""

	index = 0
	phrases = []

	arguments = defaultdict(str)

	if type(arguments_types) == str:
		phrases_tree = [{"S": phrases_tree}]
		phrases_tags = [phrases_tags]
		arguments_types = [arguments_types]

		return get_sub_phrases(phrases_tree, phrases_tags, arguments_types)

	# Moving over the sub-phrases list in the current phrases tree
	for sub_phrases_trees in phrases_tree:
		if type(sub_phrases_trees) != str:

			# Moving over the sub-phrases trees in the current sub-phrases list
			for tag, sub_phrases_tree in sub_phrases_trees.items():

				# Continue only if all the wanted phrases haven't found yet
				if index < len(phrases_tags) and (not arguments_types or index < len(arguments_types)):
					# Checking if this is a complex tag (a list)
					if type(phrases_tags[index]) == list:

						# Checking if the suitable argument is also complex of simple
						if arguments_types and type(arguments_types[index]) == list:
							inner_phrases, temp_arguments = get_sub_phrases(sub_phrases_tree, phrases_tags[index], arguments_types[index])
						else:
							inner_phrases, temp_arguments = get_sub_phrases(sub_phrases_tree, phrases_tags[index])

						if inner_phrases != []:
							if arguments_types:
								if temp_arguments:
									arguments.update(temp_arguments)
								else:
									arguments[arguments_types[index]] = clean_argument(get_phrase(sub_phrases_tree))
							phrases.append(inner_phrases)
							index += 1
						else:
							arguments = defaultdict(str)
							phrases = []

							if arguments_types:
								return phrases, arguments
							else:
								return phrases, None
					else:
						# This is a simple tag (string)

						value = None
						temp_tag = phrases_tags[index]
						if "_" in phrases_tags[index]:
							temp_tag, value = phrases_tags[index].split("_")

						if tag == temp_tag and (not value or value == sub_phrases_tree[0]):
							if arguments_types and arguments_types[index]:
								arguments[arguments_types[index]] = clean_argument(get_phrase(sub_phrases_tree))
							phrases.append({phrases_tags[index]: sub_phrases_tree})
							index += 1
						else:
							arguments = defaultdict(str)
							phrases = []

							if arguments_types:
								return phrases, arguments
							else:
								return phrases, None

				else:
					break

	if len(phrases_tags) != len(phrases):
		phrases = []
		arguments = defaultdict(str)

	if arguments_types:
		return phrases, arguments
	else:
		return phrases, None

def detect_comlex_subcat(sent):
	"""
	Detects the comblex sub-categorization of the given sentence
	:param sent: a sentence string
	:return: a list of arguments dictionaries with values that are relevant to each founded subcat
	"""

	predictor = ConstituencyParserPredictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
	phrase_tree = predictor.predict(sent)['trees']

	# Moving over each line in the input file
	# Spacing up all the opening\closing brackets
	temp_splitted_line = phrase_tree.replace("(", " ( ").replace(")", " ) ").replace(") \n", ")\n").replace("\"", "").split(' ')
	splitted_line = []

	for i in range(len(temp_splitted_line)):
		if temp_splitted_line[i] != '':
			splitted_line.append(temp_splitted_line[i].replace('\n', ''))

	new_sent = splitted_line

	# Translates the phrases tree from string to dictionary
	_, phrases_tree = process_phrases_tree(new_sent, 1)

	default_arguments = defaultdict(str)
	possible_arguments = []

	# Use the first NP, VP pair that was found in the phrases tree
	if "S" not in phrases_tree.keys():
		return []

	np_vp_phrases_trees, _ = get_sub_phrases(phrases_tree["S"], ["NP", "VP"], ["NP", "VP"])

	if np_vp_phrases_trees != []:
		np_phrase_tree = np_vp_phrases_trees[0]
		vp_phrase_tree = np_vp_phrases_trees[1]

		wordnet_lemmatizer = WordNetLemmatizer()
		default_arguments["verb"] = wordnet_lemmatizer.lemmatize(get_phrase(vp_phrase_tree).split(" ")[0], 'v')
		default_arguments["subject"] = get_phrase(np_phrase_tree)

		complex_table = get_comlex_table()

		for subcat_info in complex_table:
			subcat, tags_phrases, suitable_arguments = subcat_info

			# Even if the suitable subcat was found, a general case may also work
			_, founded_arguments = get_sub_phrases(vp_phrase_tree["VP"][1:], tags_phrases, suitable_arguments)

			# Checking if a suitable subcat was found
			if list(founded_arguments.keys()) != []:
				default_arguments["subcat"] = subcat

				# Adding the updated arguments to the possible arguments list
				temp_arguments = default_arguments.copy()
				temp_arguments.update(founded_arguments)

				if "sbar" in temp_arguments.keys() and not temp_arguments["sbar"].lower().startswith("that "):
					temp_arguments["sbar"] = "that " + temp_arguments["sbar"]

				possible_arguments.append(temp_arguments)

		# NOM-INTRANS- always suitable subcat
		default_arguments["subcat"] = "NOM-INTRANS"
		possible_arguments.append(default_arguments.copy())

		# NOM-INTRANS-RECIP- always suitable subcat in case of plural subject NP
		if inflect_engine.singular_noun(get_phrase(np_phrase_tree)):
			default_arguments["subcat"] = "NOM-INTRANS-RECIP"
			possible_arguments.append(default_arguments.copy())

	return possible_arguments


def process_a_sentence(sent):
	"""
	Processes a sentence, returns its relevant arguments
	:param sent: the sentence that was processed
	:return: all the possible founded arguments of the verb in the sentence (as a list of dictionaries)
	"""

	# Replacing the first upper letter only if the word isn't a name of something (using NER from spacy)
	dependency = get_dependency(sent)
	if dependency != [] and dependency[0][-2] == "":
		sent = sent[0].lower() + sent[1:]

	possible_arguments = detect_comlex_subcat(sent)

	seperate_line_print(possible_arguments)

	return possible_arguments

def build_pre_nom(pattern, arguments):
	"""
	Builds the pre nominalization sentence
	:param pattern: a dictionary of roles and their types
	:param arguments: an arguments dictionary, contains some arguments of a certain nominalization
	:return: the string before the nominalization (by default it must be in the next order- subject > indobject > object)
	"""

	# An argument appears in the pattern, but it wasn't found in the sentence (=> problem)
	if ("subject" not in arguments.keys() and "subject" in pattern.keys()) or \
			("object" not in arguments.keys() and "object" in pattern.keys()) or \
			("ind-object" not in arguments.keys() and "ind-object" in pattern.keys()):
		return ""

	pre_nom = ""

	if "subject" in pattern.keys():
		if pattern["subject"] == "DET-POSS":
			pre_nom += arguments["subject"] + "'s "
		elif pattern["subject"] == "N-N-MOD":
			pre_nom += det + " " + arguments["subject"] + " "

	if "ind-object" in pattern.keys():
		if pattern["ind-object"] == "DET-POSS":
			pre_nom += arguments["ind-object"] + "'s "
		elif pattern["ind-object"] == "N-N-MOD":
			if pre_nom == "":
				pre_nom += det + " " + arguments["ind-object"] + " "
			else:
				pre_nom += arguments["ind-object"] + " "

	if "object" in pattern.keys():
		if pattern["object"] == "DET-POSS":
			pre_nom += arguments["object"] + "'s "
		elif pattern["object"] == "N-N-MOD":
			if pre_nom == "":
				pre_nom += det + " " + arguments["object"] + " "
			else:
				pre_nom += arguments["object"] + " "

	if "comp-ing" in pattern.keys():
		if pattern["comp-ing"] == "DET-POSS":
			pre_nom += arguments["comp-ing"] + "'s "
		elif pattern["comp-ing"] == "N-N-MOD":
			if pre_nom == "":
				pre_nom += det + " " + arguments["comp-ing"] + " "
			else:
				pre_nom += arguments["comp-ing"] + " "

	# Adding the adverb as adj if it is eval-adv (ADJP)
	if "adverb" in pattern.keys() and pattern["adverb"] == "ADJP":
		if pre_nom == "":
			pre_nom = det + " "

		adj = get_adj(arguments["adverb"])

		if adj:
			pre_nom += adj + " "

	if pre_nom == "":
		pre_nom = det + " "

	return pre_nom

def clean_sentence(sent):
	"""
	Cleans the sentence from mistakes with pronouns and more
	:param sent: a sentnece of words
	:return: the cleaned sentence (or None = deleted)
	"""

	pronoun_dict = get_pronoun_dict()

	# Double determiners
	sent = sent.replace(" " + det + " the ", " the ").\
				replace(" " + det + " The ", " the ").\
		   		replace(" " + det + " A ", " a ").\
		   		replace(" " + det + " a ", " a ").\
		   		replace(" " + det + " an ", " an ").\
		   		replace(" " + det + " An ", " an ")

	# Pronouns
	# Translating the base form of the pronoun to the suitable form according to the context
	for pronoun, forms_list in pronoun_dict.items():
		# Pronouns and other forms + "'s"
		sent = sent.replace(" " + pronoun + "'s ", " " + forms_list[0] + " ").\
					replace(" " + pronoun[0].upper() + pronoun[1:] + "'s ", " " + forms_list[0] + " ")

		sent = sent.replace(" " + forms_list[1] + "'s ", " " + forms_list[0] + " "). \
					replace(" " + forms_list[1][0].upper() + forms_list[1][1:] + "'s ", " " + forms_list[0] + " ")

		# Pronouns + determiner
		sent = sent.replace(" " + det + " " + forms_list[0] + " ", " " + forms_list[0] + " "). \
					replace(" " + det + " " + forms_list[0].upper() + forms_list[0][1:] + " ", " " + forms_list[0] + " ")

		if " " + det + " " + pronoun + " " in sent or " " + det + " " + pronoun[0].upper() + pronoun[1:] + " " in sent:
			return None

	sent = sent.replace(" i ", " I ")

	return sent

def pattern_to_sent(nominalization, pattern, arguments):
	"""
	Translates a single pattern into a sentence\s, using context arguments
	:param nominalization: the nominalization word
	:param pattern: a pattern, that should be translated (as a dictionary)
	:param arguments: a dictionary of context arguments, that helps in the translation
	:return: list of suitable nominal sentences for the given data
	"""

	sentences = []

	if should_print: print(pattern)

	# Getting the sentence string before the nominalization
	pre_nom = build_pre_nom(pattern, arguments)

	# Adding the nominalization
	sentence = pre_nom + nominalization

	# Getting all the prepositions which appear in the pattern
	# Those prepositions will appear after the nominalization in any order
	post_preps = []
	for subentry, option in pattern.items():
		if type(option) == str and option.startswith("PP-"):

			# Fixing msitakes of pronouns after prepositions (only for the resulted sentence)
			arg = arguments[subentry]
			for pronoun, forms_list in get_pronoun_dict().items():
				if arguments[subentry] == pronoun or (pronoun[0].upper() + pronoun[1:]) == arguments[subentry]:
					arg = forms_list[1]

			post_preps.append([option.replace("PP-", "").lower(), arg])

		elif subentry in ["pval", "pval1", "pval2", "pval-ing", "pval-comp-ing", "adjective", "to-inf", "pval-to-inf"]:
			splitted = arguments[subentry].split(" ")

			if option == "pval-nom" or option == "pval1-nom" or option == "pval2-nom":
				option = pattern[option]
				arguments[subentry] = option + " " + " ".join(splitted[1:])

			post_preps.append([option.lower(), " ".join(splitted[1:])])

		elif subentry == "sbar":
			if arguments[subentry].startswith("that"):
				post_preps.append([arguments[subentry]])
			else:
				post_preps.append(["that", arguments[subentry]])

		elif subentry == "ing":
			post_preps.append([option.lower(), arguments[subentry]])

		elif subentry == "adverb" and option == "ADVP":
			post_preps.append([arguments["adverb"]])

	# Finally, adding the relevant prepositions from the pattern (in any order)
	for preps_order in itertools.permutations(post_preps, len(post_preps)):
		temp_sentence = sentence
		for prep in preps_order:
			if len(prep) == 1:
				temp_sentence += " " + prep[0]
			else:
				temp_sentence += " " + prep[0] + " " + prep[1]

		sentences.append(temp_sentence)

	# Cleaning the resulted sentences
	new_sentences = []
	for i in range(len(sentences)):
		sentences[i] = " " + sentences[i] + " "
		temp = clean_sentence(sentences[i])

		if temp:
			sentences[i] = temp
			sentences[i] = sentences[i][1:-1] # Without the added spaces
			sentences[i] = sentences[i][0].upper() + sentences[i][1:] + "."
			new_sentences.append(sentences[i])

	return new_sentences

def verbal_to_nominal(nomlex_entries, sent):
	"""
	Translates a verbal sentence into a nominal sentence, using nominalizations
	Assumption- the sentence contain only one verb
	:param nomlex_entries: the json formatted data of nomlex lexicon
	:param sent: a given verbal sentence
	:return: a list of nominal suitable sentences for the given sentence
	"""

	# Getting the arguments for the verb in the sentence (= processing the sentence)
	possible_arguments = process_a_sentence(sent)

	nom_sentences = []
	founded_noms = []

	# There may be many possible arguments dictionaries
	for arguments in possible_arguments:
		# Getting the relevant nominalization entries according to the verb that we found
		relevant_entries = get_nomlex_entries(nomlex_entries, arguments["verb"])

		# Extracting all the suitable nominalization patterns
		nom_patterns = extract_nom_patterns(relevant_entries, arguments["subcat"])

		# Creating all the nominalization suitable sentences for the given sentence
		for nominalization, patterns in nom_patterns.items():
			if nominalization not in founded_noms:
				for pattern in patterns:
					nom_sentences += pattern_to_sent(get_all_of_noms(nomlex_entries)[nominalization], pattern, arguments)
					founded_noms.append(nominalization) # The first suitable arguments list is preferable

	return list(set(nom_sentences))





############################################## Extracting Arguments ##############################################

def get_dependency(sent):
	"""
	Returns the dependency tree of a given sentence
	:param sent: a string sentence
	:return: the dependency tree of the sentence (a list of tuples)
	"""

	dep = []

	sentence_info = nlp(sent)
	for word_info in sentence_info:
		head_id = str(word_info.head.i + 1)  # we want ids to be 1 based
		if word_info == word_info.head:  # and the ROOT to be 0.
			assert (word_info.dep_ == "ROOT"), word_info.dep_
			head_id = "0"  # root

		str_sub_tree = " ".join([node.text for node in word_info.subtree])
		dep.append(
			(word_info.i + 1, str(word_info.text), str(word_info.lemma_),
			 str(word_info.tag_), str(word_info.pos_), int(head_id),
			 str(word_info.dep_), str(word_info.ent_iob_), str(word_info.ent_type_), str_sub_tree))

	return dep

def fix_ud_links(ud_links, option):
	"""
	Fixes some unvirsal dependencies links recursively
	For example, making rules that ends with "_" for specific (according to the given  option)
	:param ud_links: a list of the universal dependencies (each x in the list can be also a list or a string)
	:param option: a string which represent an option (like preposition string and so on)
	:return: The fixed universal dependencies list
	"""

	new_ud_links = []

	for i in range(len(ud_links)):
		# Go recursively if x is a list
		if type(ud_links[i]) == list:
			new_ud_links.append(fix_ud_links(ud_links[i], option))
		elif type(ud_links[i]) == str and ud_links[i].endswith("_"):
			new_ud_links.append(ud_links[i] + option.lower())
		else:
			new_ud_links.append(ud_links[i])

	return new_ud_links

def pattern_to_UD(pattern):
	"""
	Translates a pattern into universal dependency sequence
	:param pattern: a given pattern (dictionary subentry: option)
	:return: a suitable sequence of universal dependency links (all the links are in outside direction)
	"""

	pattern_UD = defaultdict(list)

	if should_print: print(pattern)

	subentries_table = get_subentries_table()
	subentries_types = [i[0] for i in subentries_table]

	for subentry, option in pattern.items():
		if subentry in subentries_types and option != "NOM":
			ud_links = subentries_table[subentries_types.index(subentry)][1]

			# Dealing with preposition options that starts with "PP-"
			temp_option = option
			if option.startswith("PP-"):
				option = option.replace("PP-", "")
				temp_option = "PP-"

			# The ud links also depends on the option of the subentry
			if type(ud_links) == dict:
				ud_links = ud_links[temp_option]

			# Making some links more specific (links that ends with "_")
			pattern_UD[subentry] = fix_ud_links(ud_links, option)

	return pattern_UD

def clean_argument(argument):
	"""
	Cleans the argument from mistakes with pronouns and more
	:param argument: a word or sentnece, which is an argument or the nominalization
	:return: the cleaned argument
	"""

	pronoun_dict = get_pronoun_dict()
	arg = argument

	# Translating other forms of the pronoun to the base form
	for pronoun, forms_list in pronoun_dict.items():
		if argument.lower() in forms_list:
			arg = pronoun

	# Deleting the ending " 's" in case that the argument was DET-POSS
	if arg.endswith(" 's"):
		arg = arg[:-3]

	return arg

def extract_argument(dep_tree, dep_links, dep_curr_index):
	"""
	A recursive function that finds an argument acording to the given dependency links
	:param dep_tree: the dependency tree of the sentence
	:param dep_links: a list of dependency links
	:param dep_curr_index: the current index in the tree dependency
	:return: the suitable arguments that we get if we follow the given links backwards from the current index
	"""

	# Stop Conditions
	if dep_links == []:
		if dep_curr_index == -1:
			return []
		else:
			arg = dep_tree[dep_curr_index][9]
			return [(dep_tree[dep_curr_index][0], arg)]

	if dep_curr_index == -1:
		return []

	arguments = []

	if type(dep_links[0]) == list:
		if extract_argument(dep_tree, dep_links[0], dep_curr_index) == []:
			return []

		arguments += extract_argument(dep_tree, dep_links[1:], dep_curr_index)

	for i in range(len(dep_tree)):
		temp_arguments = []

		# Checking if the node links to the current node
		if dep_tree[i][5] - 1 == dep_curr_index:
			# Checking that the link type is right
			if "_" in dep_links[0]:
				splitted = dep_links[0].split("_")

				# link type + exact word as type_word
				if len(splitted) == 2:
					if dep_tree[i][6] == splitted[0] and dep_tree[i][1] == splitted[1]:
						temp_arguments += extract_argument(dep_tree, dep_links[1:], i)

				# link type + ending as type__ending
				elif len(splitted) == 3:
					if dep_tree[i][6] == splitted[0] and dep_tree[i][1].endswith(splitted[2]):
						temp_arguments += extract_argument(dep_tree, dep_links[1:], i)

			# Only a link type
			elif dep_tree[i][6] == dep_links[0]:
				temp_arguments += extract_argument(dep_tree, dep_links[1:], i)

			# We want to remember the prep position, because each preposition can relate only for one argument
			if type(dep_links[0]) == str and dep_links[0].startswith("prep_"):
				for _, arg in temp_arguments:
					arguments.append((i, arg))
			else:
				arguments += temp_arguments

	return arguments

def get_arguments(dependency_tree, nom_entry, nom_index):
	"""
	Returns the all the possible arguments for a specific nominalization in a sentence with the given dependency tree
	:param dependency_tree: a universal dependency tree (a list of tuples)
	:param nom_entry: the information inside a specific nominalization entry in the NOMLEX lexicon
	:param nom_index: the index of the nominalization in the given dependency tree
	:return: a list of dictionaries (in the list all the possible arguments, dictionary for each possible set of arguments)
	"""

	# Getting the nominalization patterns
	patterns = get_nom_patterns(nom_entry)

	total_arguments = []
	subentries_types = [i[0] for i in get_subentries_table()]
	comlex_subcats = [i[0] for i in get_comlex_table()]

	# Moving over all the possible patterns for the given nominalization
	# Trying to extract all the possible arguments for that nominalization
	for pattern in patterns:
		if pattern["subcat"] in comlex_subcats + ["NOM-INTRANS", "NOM-INTRANS-RECIP"]:

			if "pval" in pattern.keys() and pattern["pval"] == "pval-nom":
				pattern["pval"] = pattern["pval-nom"]
				pattern.remove("pval-nom")

			if "ind-object" in pattern.keys() and pattern["ind-object"] == "pval1-nom":
				pattern["ind-object"] = pattern["pval1-nom"]
				pattern.remove("pval1-nom")

			if "pval2" in pattern.keys() and pattern["pval2"] == "pval2-nom":
				pattern["pval2"] = pattern["pval2-nom"]
				pattern.remove("pval2-nom")

			# Translating the pattern into universal dependencies sequence
			pattern_UD = pattern_to_UD(pattern)

			# Initiate the current arguments dictionary
			curr_arguments = defaultdict(tuple)
			curr_arguments["verb"] = (-1, -1, nom_entry["VERB"])
			curr_arguments["subcat"] = (-1, -1, pattern["subcat"])

			# Is the nominalization itself has a role in the sentence, rather than replacing the verb (= action)
			if "SUBJECT" in nom_entry["NOM-TYPE"].keys():
				curr_arguments["subject"] = (-1, -1, dependency_tree[nom_index][1])
			elif "OBJECT" in nom_entry["NOM-TYPE"].keys():
				curr_arguments["object"] = (-1, -1, dependency_tree[nom_index][1])

			curr_arguments_list = [curr_arguments]
			new_curr_arguements_list = curr_arguments_list.copy()

			# Looking for each argument (the order is important, because subject > indobject > object and not otherwise)
			for subentry in subentries_types:
				if subentry in pattern_UD.keys():
					dep_links = pattern_UD[subentry]
					possible_arguments = extract_argument(dependency_tree, dep_links, nom_index)

					# Checking all the possible arguments that were extracted for the current subentry
					if possible_arguments != []:
						for arguments in curr_arguments_list:
							for index, arg in possible_arguments:
								temp_arguments = arguments.copy()

								# Translate adjective to adverb if needed
								if subentry == "adverb" and pattern[subentry] == "ADJP":
									arg = get_adv(arg)
								elif subentry == "adjective":
									arg = pattern[subentry] + " " + arg
								else:
									arg = clean_argument(arg)

								curr_indexes = [i for i, _, _ in temp_arguments.values()]
								rel_indexes = [i for _, i, _ in temp_arguments.values()]
								if index not in curr_indexes:
									if subentry in ["subject", "ind-object", "object"]:
										if pattern[subentry].startswith("PP-"):
											temp_arguments[subentry] = (index, -1, arg)
										elif index > max(rel_indexes):
											temp_arguments[subentry] = (index, index, arg)
									else:
										temp_arguments[subentry] = (index, index, arg)

									new_curr_arguements_list.append(temp_arguments)

						curr_arguments_list = new_curr_arguements_list.copy()

			# Add only the full lists of arguments that were found
			for new_curr_arguments in new_curr_arguements_list.copy():
				if set(pattern.keys()) == set(new_curr_arguments.keys()):
					if pattern["subcat"] != "NOM-INTRANS-RECIP" or \
						(new_curr_arguments["subject"] != () and inflect_engine.singular_noun(new_curr_arguments["subject"][2])):
						total_arguments.append(new_curr_arguments)

	return total_arguments

def extract_arguments(nomlex_entries, sent):
	"""
	Extracts the arguments of the nominalizations in the given sentence
	:param nomlex_entries: NOMLEX entries (a dictionary nom: ...)
	:param sent: a given sentence (string)
	:return: a dictionary of lists of dictionaries
			 dictionary of each founded nominalization (nom, index) -> list of each suitable pattern -> dictionary of arguments
	"""

	sent = sent.replace("’", "'")

	# Replacing the first upper letter only if the word isn't a name of something (using NER from spacy)
	dependency = get_dependency(sent)
	if dependency != [] and dependency[0][-2] == "":
		sent = sent[0].lower() + sent[1:]

	# Getting the dependency tree of the sentence
	dependency_tree = get_dependency(sent)

	# Finding all the nominalizations in the sentence
	noms = []
	all_noms = get_all_of_noms(nomlex_entries)
	for i in range(len(dependency_tree)):
		for nom, clean_nom in all_noms.items():
			if dependency_tree[i][2] == clean_nom:
				noms.append((nom, i))

	# Moving over all the nominalizations
	nom_args = {}
	for nom, nom_index in noms:
		# Getting the suitable nominalization entry
		nom_entry = nomlex_entries[nom]

		# Getting all the possible arguments
		arguments_list = get_arguments(dependency_tree, nom_entry, nom_index)

		# Finding the maximum number of arguments that were extracted
		best_num_of_args = 0
		for args in arguments_list:
			if len(args.keys()) > best_num_of_args:
				best_num_of_args = len(args.keys())

		# Add all the "best arguments" that were extracted (best = maximum number of arguments)
		best_args = []
		best_args_items = [] # List of all the items that were extracted (for singularity)
		for args in arguments_list:
			# Checking the number of arguments in args, and singularity
			if len(args.keys()) == best_num_of_args and args.items() not in best_args_items:
				new_args = defaultdict(str)
				for role, (_, _, arg) in args.items():
					new_args[role] = arg

				best_args.append(new_args)
				best_args_items.append(args.items())

		nom_args.update({(nom, nom_index): best_args})

	return nom_args




############################################### Matching Arguments ###############################################

def match_arguments(nomlex_entries, verbal_sentence, nominal_sentence):
	"""
	Finds all the exact matching between the arguments of the verb and nominalizations
	The verbal sentence defines that rule that needed to be found in the nominal sentences
	:param nomlex_entries: NOMLEX entries (a dictionary nom: ...)
	:param verbal_sentence: a simple sentence with a main verb
	:param nominal_sentence: a complex sentence that may include nominalizations
	:return: a list of the nominalizations in the nominal sentence that their arguments match the arguments of the main verb in the verbal sentences
	"""

	# Getting the arguments for the verb in the sentence (= processing the sentence)
	possible_verb_arguments = process_a_sentence(verbal_sentence)

	matching_noms = {}

	# There may be many possible arguments dictionaries
	for verb_arguments in possible_verb_arguments:
		temp_verb_arguments = verb_arguments.copy()

		# Removing the preposition word (in, on, about, ...) from preposition arguments
		for prep in ["pval", "pval2", "pval-ing"]:
			if prep in temp_verb_arguments.keys() and len(temp_verb_arguments[prep]):
				temp_verb_arguments[prep] = " ".join(temp_verb_arguments[prep].split(" ")[1:])

		# We want to match the lower case of the arguments
		for argument in temp_verb_arguments.keys():
			temp_verb_arguments[argument] = temp_verb_arguments[argument].lower()

		# Removing subcat argument
		temp_verb_arguments.pop("subcat")
		saved_verb_arguments = temp_verb_arguments.copy()

		noms_arguments = extract_arguments(nomlex_entries, nominal_sentence)

		# Checking each nominalization that was found
		for nom, possible_nom_arguments in noms_arguments.items():
			current_matching_patterns = []

			# And all its possible arguments that were extracted
			for nom_arguments in possible_nom_arguments:
				temp_nom_arguments = nom_arguments.copy()

				# Removing the preposition word (in, on, about, ...)
				for prep in ["pval", "pval2", "pval-ing"]:
					if prep in temp_nom_arguments.keys() and len(temp_nom_arguments[prep]):
						temp_nom_arguments[prep] = " ".join(temp_nom_arguments[prep].split(" ")[1:])

				# We want to match the lower case of the arguments
				for argument in temp_nom_arguments.keys():
					temp_nom_arguments[argument] = temp_nom_arguments[argument].lower()

				# Removing subcat argument
				temp_nom_arguments.pop("subcat")

				# Removing the subject if it is the nominalization
				if "subject" in temp_nom_arguments and "subject" in temp_verb_arguments and temp_nom_arguments["subject"] == nom[0]:
					temp_nom_arguments.pop("subject")
					temp_verb_arguments.pop("subject")

				# Removing the object if it is the nominalization
				elif "object" in temp_nom_arguments and "object" in temp_verb_arguments and temp_nom_arguments["object"] == nom[0]:
					temp_nom_arguments.pop("object")
					temp_verb_arguments.pop("object")

				# Comparing between the current pair of arguments
				if temp_verb_arguments == temp_nom_arguments:
					current_matching_patterns.append(nom_arguments)

				temp_verb_arguments = saved_verb_arguments.copy()

			if current_matching_patterns != []:
				matching_noms.update({nom: (verb_arguments, current_matching_patterns)})

	return matching_noms




############################################### Loading and Saving ###############################################

def load_json_data(json_file_name):
	"""
	Loads the data from a json file
	:param json_file_name: The name of the file that needed to be saved
	:return: The the json data (basically a dictionary object)
	"""

	with open(json_file_name) as inputfile:
		data = json.load(inputfile)

	return data




################################################### Utilities ####################################################

def get_best_word(word, possible_list, preferable_endings):
	"""
	Returns the most relevant word in the possible list to the given word
	The most relevant is a word that starts the same as the given word
	Also prefer words according to the given list of preferable endings
	:param word: a word
	:param possible_list: a list of words
	:param preferable_endings: a list of strings of preferable endings of the wanted word
	:return: the most relevant word to the given word
	"""

	if possible_list == []:
		return None

	best_word = possible_list[0]
	best_subword_length = 0
	for possible_word in possible_list:
		i = 0
		while i < len(word) and i < len(possible_word) and possible_word[i] == word[i]:
			i += 1

		i -= 1
		if i >= best_subword_length or (i == best_subword_length and any([possible_word.endswith(end) for end in preferable_endings])):
			best_subword_length = i
			best_word = possible_word

	return best_word

def get_adj(word):
	"""
	Returns the best adjective that relates to the given word (if no adjective was found, None is returned)
	:param word: a word
	:return: an adjective that is most relevant to the given word, or the given word (if no adjective was found)
	"""

	possible_adj = []
	for ss in wn.synsets(word):
		for lemmas in ss.lemmas():  # all possible lemmas
			for ps in lemmas.pertainyms():  # all possible pertainyms (the adjectives of a noun)
				possible_adj.append(ps.name())

	best_adj = get_best_word(word, possible_adj, ["able", "ible", "al", "ful", "ic", "ive", "less", "ous"])

	if best_adj:
		return best_adj

	return word

def get_adv(word):
	"""
	Returns the best adverb that relates to the given word (if no adverb was found, None is returned)
	:param word: a word
	:return: an adverb that is most relevant to the given word, or the given word (if no adverb was found)
	"""

	possible_adv = []
	for synset in list(wn.all_synsets('r')):
		if get_adj(synset.lemmas()[0].name()) == word:
			possible_adv.append(synset.lemmas()[0].name())

	best_adv = get_best_word(word, possible_adv, ["ly", "ward", "wise"])

	if best_adv:
		return best_adv

	return word


def seperate_line_print(input_to_print):
	if type(input_to_print) == list:
		for x in input_to_print:
			if should_print: print(x)
	elif type(input_to_print) == dict:
		for tag, x in input_to_print.items():
			if should_print: print(str(tag) + ": " + str(x))


def get_all_of_noms(nomlex_entries):
	"""
	Returns a dictionary of all the nominalizations in the given nomlex entries
	:param nomlex_entries: a dictionary of nominalizations
	:return: dictionary of nominalizations (nominalizations: nominalizations_without_numbers)
	"""

	all_noms = {}

	for nom in nomlex_entries.keys():
		all_noms.update({nom: "".join([i for i in nom if not i.isdigit()])})

	return all_noms




###################################################### Main ######################################################

def main(arguments):
	"""
	The main function
	:param arguments: the command line arguments
	:return: None
	"""

	# Extraction of patterns- from verbal sentence to nominal sentence
	if arguments[0] == "-patterns" and len(arguments) == 3:
		json_file_name = arguments[1]
		verbal_sent = arguments[2]

		nomlex_entries = load_json_data(json_file_name)
		seperate_line_print(verbal_to_nominal(nomlex_entries, verbal_sent))

	# Extraction of arguments- from nominal sentence to verbal sentence
	elif arguments[0] == "-args" and len(arguments) == 3:
		json_file_name = arguments[1]
		nominal_sent = arguments[2]

		nomlex_entries = load_json_data(json_file_name)
		seperate_line_print(extract_arguments(nomlex_entries, nominal_sent))

	# Matching arguments extracted in verbal and nominal sentences
	elif arguments[0] == "-match" and len(arguments) == 4:
		json_file_name = arguments[1]
		verbal_sent = arguments[2]
		nominal_sent = arguments[3]

		nomlex_entries = load_json_data(json_file_name)
		seperate_line_print(match_arguments(nomlex_entries, verbal_sent, nominal_sent))

if __name__ == '__main__':
	"""
	Command line arguments-
		 -patterns json_file_name verbal_sentence
		 -args json_file_name noninal_sentence
		 -match json_file_name verbal_sentence nominal_sentence
	"""
	import sys

	main(sys.argv[1:])