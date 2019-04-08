from collections import defaultdict
import spacy
import inflect
inflect_engine = inflect.engine()
nlp = spacy.load('en_core_web_sm')

from DictsAndTables import get_comlex_table, get_subentries_table, get_pronoun_dict, should_print, get_all_of_noms, get_adv
from ExtractNomlexPatterns import get_nom_patterns

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

	pattern_UD_list = [defaultdict(list)]

	if should_print: print(pattern)

	subentries_table = get_subentries_table()
	subentries_types = [i[0] for i in subentries_table]

	for subentry, option in pattern.items():
		if subentry in subentries_types and option != "NOM":
			ud_links_list = subentries_table[subentries_types.index(subentry)][1]

			last_pattern_UD_list = pattern_UD_list.copy()
			pattern_UD_list = []

			for ud_links in ud_links_list:
				# Dealing with preposition options that starts with "PP-"
				temp_option = option
				if option.startswith("PP-"):
					option = option.replace("PP-", "")
					temp_option = "PP-"
				elif any(i in option for i in [" what", " whether", " how"]):
					temp_option = option.split(" ")[-1]
					option = " ".join(option.split(" ")[:-1])

				# The ud links also depends on the option of the subentry
				if type(ud_links) == dict:
					ud_links = ud_links[temp_option]

				# Making some links more specific (links that ends with "_")
				for pattern_UD in last_pattern_UD_list:
					pattern_UD[subentry] = fix_ud_links(ud_links, option)
					pattern_UD_list.append(pattern_UD.copy())

	return pattern_UD_list

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

	arg = arg.replace(" 's", "'s")
	arg = arg.replace("s '", "s'")

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

			# Translating the pattern into universal dependencies sequences (can be more than one possibel sequence)
			pattern_UD_list = pattern_to_UD(pattern)

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
				# Moving over each universal dependencies sequence that was found
				for pattern_UD in pattern_UD_list:
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

def extract_patterns_from_nominal(nomlex_entries, sent):
	"""
	Extracts the arguments of the nominalizations in the given sentence
	:param nomlex_entries: NOMLEX entries (a dictionary nom: ...)
	:param sent: a given sentence (string)
	:return: a dictionary of lists of dictionaries
			 dictionary of each founded nominalization (nom, index) -> list of each suitable pattern -> dictionary of arguments
	"""

	sent = sent.replace("’", "'")

	# Getting the dependency tree of the sentence
	dependency_tree = get_dependency(sent)

	# Replacing the first upper letter only if the word isn't a name of something (using NER from spacy)
	# Replacing in the tree, and not in the sentence
	if dependency_tree != [] and dependency_tree[0][-2] == "":
		dependency_tree[0] = (dependency_tree[0][0], dependency_tree[0][1].lower(), dependency_tree[0][2],
							  dependency_tree[0][3], dependency_tree[0][4], dependency_tree[0][5],
							  dependency_tree[0][6], dependency_tree[0][7], dependency_tree[0][8],
							  dependency_tree[0][9][0].lower() + dependency_tree[0][9][1:])

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