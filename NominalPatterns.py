from collections import defaultdict
import spacy
import inflect
import copy
inflect_engine = inflect.engine()
#spacy.prefer_gpu()
nlp = spacy.load('en_core_web_sm')

import DictsAndTables
from DictsAndTables import subentries_table, pronoun_dict, special_preps_dict, comlex_subcats,\
						   get_adv, replace_empty_list
from ExtractNomlexPatterns import get_nom_patterns



def get_dependency(sent):
	"""
	Returns the dependency tree of a given sentence
	:param sent: a string sentence
	:return: the dependency tree of the sentence (a list of tuples)
	"""

	dep = []

	# Here, the dependency tree is created using Spacy Package
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
	This is a recursive function that fixes some unvirsal dependencies links
	For example, making rules that ends with "_" more specific (according to the given option, or special preps dict)
	:param ud_links: a list of the universal dependencies (each x in the list can be also a list or a string)
	:param option: a string which represent an option (like preposition string and so on)
	:return: The fixed universal dependencies list
	"""

	new_ud_links = []

	for i in range(len(ud_links)):
		# Go recursively if x is a list
		if type(ud_links[i]) == list:
			new_ud_links.append(fix_ud_links(ud_links[i], option))

		# "prep_" case
		elif type(ud_links[i]) == str and ud_links[i].endswith("_"):
			if option.lower() in special_preps_dict.keys():
				# Replacing prep_ with special links for a specific preposition (with more than one word)
				if i + 1 < len(ud_links) and type(ud_links[i + 1]) == list:
					new_ud_links += copy.deepcopy(special_preps_dict[option.lower()][1][1])
				else:
					new_ud_links += copy.deepcopy(special_preps_dict[option.lower()][1][0])

				# Adding the next dependency links in the right place aftr the new added links
				replace_empty_list(new_ud_links, ud_links[i + 1:])

				return new_ud_links
			else:
				new_ud_links.append(ud_links[i] + option.lower()) # Just adding the specific preposition type

		# Otherwise
		else:
			new_ud_links.append(ud_links[i])

	return new_ud_links

def pattern_to_UD(pattern):
	"""
	Translates a pattern into universal dependency sequence
	Currently the dependencies tags are the default dependency tags from Spacy (named clearNLP)
	Hopefully, in the future we will use UD
	:param pattern: a given pattern (dictionary {subentry: option})
	:return: a suitable sequence of universal dependency links (all the links are in outside direction)
	"""

	pattern_UD_list = [defaultdict(list)]

	tmp_pattern = pattern.copy()

	# Fixing special arguments for nominalization
	if "pval" in tmp_pattern.keys() and tmp_pattern["pval"] == "pval-nom":
		tmp_pattern["pval"] = tmp_pattern["pval-nom"]
		tmp_pattern.remove("pval-nom")

	if "ind-object" in tmp_pattern.keys() and tmp_pattern["ind-object"] == "pval1-nom":
		tmp_pattern["ind-object"] = tmp_pattern["pval1-nom"]
		tmp_pattern.remove("pval1-nom")

	if "pval2" in tmp_pattern.keys() and tmp_pattern["pval2"] == "pval2-nom":
		tmp_pattern["pval2"] = tmp_pattern["pval2-nom"]
		tmp_pattern.remove("pval2-nom")

	if DictsAndTables.should_print and DictsAndTables.should_print_to_screen:
		print(dict(tmp_pattern))

	subentries_types = [i[0] for i in subentries_table]

	for subentry, option in tmp_pattern.items():
		if subentry in subentries_types and option != "NOM":
			ud_links_list = subentries_table[subentries_types.index(subentry)][1]

			last_pattern_UD_list = copy.deepcopy(pattern_UD_list)
			pattern_UD_list = []

			for ud_links in ud_links_list:
				# Dealing with preposition options that starts with "PP-"
				temp_option = option
				if option.startswith("PP-"):
					option = option.replace("PP-", "")
					temp_option = "PP-"

				# Dealing with wh subentry
				elif any(i in option for i in [" what", " whether", " how"]):
					temp_option = option.split(" ")[-1]
					option = " ".join(option.split(" ")[:-1])

				# Sometimes the ud links also depends on the option of the subentry (e.g. subject)
				if type(ud_links) == dict:
					ud_links = ud_links[temp_option]
				else:
					ud_links = [ud_links]

				for x in ud_links:
					for pattern_UD in last_pattern_UD_list:
						# Making some links more specific (for example links that ends with "_")
						pattern_UD[subentry] = fix_ud_links(x, option)

						pattern_UD_list.append(pattern_UD.copy())

	return pattern_UD_list

def clean_argument(argument):
	"""
	Cleans the argument from mistakes with pronouns and more
	:param argument: a word or sentnece, which is a founded argument of a nominalization
	:return: the cleaned argument
	"""

	arg = argument

	if DictsAndTables.should_clean:
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
	The extracted argument is found using the given dependency links and the given index
	:param dep_tree: the dependency tree of the sentence
	:param dep_links: a list of dependency links
	:param dep_curr_index: the current index in the dependency tree
	:return: a list of the suitable arguments that we get if we follow the given links backwards from the current index
	"""

	# Stop Conditions
	if dep_links == []:
		if dep_curr_index == -1:
			return []

		# The argument is the appropriate sentence to the subtree rooted at the current index in the tree
		arg = dep_tree[dep_curr_index][9]

		first_index = dep_tree[dep_curr_index][0] - 1

		while dep_tree[first_index][1] != arg.split(" ")[0]:
			first_index -= 1

		return [(first_index, arg)]

	if dep_curr_index == -1:
		return []

	arguments = []

	# Always follow the first dependency link in the current dependency links list

	# The current link is a list, so continue recursively
	if type(dep_links[0]) == list:
		if extract_argument(dep_tree, dep_links[0], dep_curr_index) == []:
			return []

		arguments += extract_argument(dep_tree, dep_links[1:], dep_curr_index)

	# Otherwise, the current link is a string, so find the argument it connected to
	for i in range(len(dep_tree)):
		temp_arguments = []

		# Checking if the node links to the current node
		if dep_tree[i][5] - 1 == dep_curr_index:
			# Detecting link type
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

			# We want to remember the prep index, because each preposition can relate only for one argument
			if type(dep_links[0]) == str and dep_links[0].startswith("prep_"):
				for _, arg in temp_arguments:
					arguments.append((i, arg))
			else:
				arguments += temp_arguments

	return arguments

def get_arguments(dependency_tree, nom_entry, nom_index, patterns=None):
	"""
	Returns all the possible arguments for a specific nominalization in a sentence according to it dependency tree
	:param dependency_tree: a dependency tree (a list of tuples)
	:param nom_entry: the information inside a specific nominalization entry of NOMLEX lexicon
	:param nom_index: the index of the nominalization in the given dependency tree
	:param patterns: the already known patterns for the nominalization (list of patterns, each pattern is a dictionary), optional
	:return: a list of dictionaries (in the list all the possible arguments, dictionary for each possible set of arguments (=pattern))
	"""

	# Getting the nominalization patterns using the nomlex lexicon
	if not patterns:
		patterns = get_nom_patterns(nom_entry)

	total_arguments = []
	remembered_arguments_dict = {}

	# Moving over all the possible patterns for the given nominalization
	# Trying to extract all the possible arguments for that nominalization
	for pattern in patterns:
		# a pattern can already iclude the ud_pattern
		if type(pattern) == tuple:
			tmp_pattern = pattern[0].copy()
		else:
			tmp_pattern = pattern.copy()

		# Ignoring subcats that don't appear in our comlex table
		if "subcat" not in tmp_pattern.keys() or tmp_pattern["subcat"] in comlex_subcats:

			# a pattern can already iclude the ud_pattern
			if type(pattern) == tuple:
				pattern_UD_list = [pattern[1]]
			else:
				# Translating the pattern into dependency links sequences (can be more than one possibel sequence)
				pattern_UD_list = pattern_to_UD(tmp_pattern)

			# Initiating the current arguments dictionary
			curr_arguments = defaultdict(tuple)
			if "verb" in tmp_pattern.keys():
				curr_arguments["verb"] = (-1, -1, nom_entry["VERB"])

			if "subcat" in tmp_pattern.keys():
				curr_arguments["subcat"] = (-1, -1, tmp_pattern["subcat"])

			# Is the nominalization itself has a role in the sentence, rather than replacing the verb (= action)
			# Here we ignore cases that NOM-TYPE is SUBJECT + OBJECT or SUBJECT\OBJECT + VERB-NOM
			if list(nom_entry["NOM-TYPE"].keys()) == ["SUBJECT"] and tmp_pattern["subject"] == "NOM":
				curr_arguments["subject"] = (-1, -1, dependency_tree[nom_index][1])
			elif list(nom_entry["NOM-TYPE"].keys()) == ["OBJECT"] and tmp_pattern["object"] == "NOM":
				curr_arguments["object"] = (-1, -1, dependency_tree[nom_index][1])

			# Moving over each dependency links sequence that was found
			for pattern_UD in pattern_UD_list:
				full_arguments = True

				# Initiating the basic arguments lists (for pattern sequence)
				curr_arguments_list = [curr_arguments]

				# Looking for each argument
				for subentry in list(pattern_UD.keys()):
					dep_links = pattern_UD[subentry]

					# That dependency links looks familiar
					remembered_arguments = remembered_arguments_dict.get(str(dep_links), -1)
					if remembered_arguments == []:
						full_arguments = False
						break

				if full_arguments:
					full_arguments = False

					# The order of arguments\subentries is important, because it should be subject > indobject > object and not otherwise
					subentries_types = list(tmp_pattern.keys())
					for x in ["object", "indobject", "subject"]:
						if x in subentries_types:
							subentries_types.remove(x)
							subentries_types = [x] + subentries_types

					if "subcat" in subentries_types:
						subentries_types.remove("subcat")

					if "verb" in subentries_types:
						subentries_types.remove("verb")

					# Looking for each argument
					for subentry in subentries_types:
						if subentry in pattern_UD.keys():
							dep_links = pattern_UD[subentry]
						elif tmp_pattern[subentry] == "NOM":
							dep_links = "NOM"
						else:
							continue

						# That dependency links looks familiar
						remembered_arguments = remembered_arguments_dict.get(str(dep_links), -1)
						if remembered_arguments == -1:
							# If not, then the arguments should be extracted (for the first time, from the sentence)
							if tmp_pattern[subentry] == "NOM":
								possible_arguments = [(nom_index, dependency_tree[nom_index][1])]
							else:
								possible_arguments = extract_argument(dependency_tree, dep_links, nom_index)

							updated_possible_arguments = []

							# Fixing and cleaning specific arguments, like translating adjective to adverb if needed
							for index, arg in possible_arguments:
								if subentry == "adverb" and tmp_pattern[subentry] == "ADJP" and DictsAndTables.should_clean:
									updated_possible_arguments.append((index, get_adv(arg)))
								elif subentry == "adjective":
									updated_possible_arguments.append((index, tmp_pattern[subentry] + " " + arg))
								elif subentry == "subject":
									# The subcat "NOT-INTRANS-RECIP" is legal only when the extracted subject is plural
									if "subcat" not in tmp_pattern.keys() or tmp_pattern["subcat"] != "NOM-INTRANS-RECIP" or \
										inflect_engine.singular_noun(arg):
										updated_possible_arguments.append((index, clean_argument(arg)))
								else:
									updated_possible_arguments.append((index, clean_argument(arg)))

							possible_arguments = updated_possible_arguments

							# Saving the extracted arguments in the dictionary
							remembered_arguments_dict[str(dep_links)] = possible_arguments
						else:
							possible_arguments = remembered_arguments

						# Checking all the possible arguments that were extracted for the current subentry
						if possible_arguments != []:
							full_arguments = True
							new_arguments_list = []

							for arguments in curr_arguments_list:
								for index, arg in possible_arguments:
									temp_arguments = arguments.copy()

									is_good_arg = True

									# Avoiding from extracting multiple arguments from the same part of the sentence
									# More over, this code deals with the importance of the order of subject > ind-object > object (before the nominalization)
									curr_indexes = [i for i, _, _ in temp_arguments.values()]
									relevant_indexes = [i for _, i, _ in temp_arguments.values()]
									if index not in curr_indexes:
										if subentry in ["subject", "ind-object", "object"]:
											if tmp_pattern[subentry].startswith("PP-"):
												temp_arguments[subentry] = (index, -1, arg)
											elif relevant_indexes == [] or index > max(relevant_indexes):
												temp_arguments[subentry] = (index, index, arg)
											else:
												is_good_arg = False
										else:
											temp_arguments[subentry] = (index, index, arg)

										if is_good_arg:
											new_arguments_list.append(temp_arguments)

							curr_arguments_list = new_arguments_list
						else:
							full_arguments = False
							break # Ignoring not full lists of arguments

					# Add the current arguments lists only if all the relevant arguments were found
					if full_arguments:
						# Add only the full lists of arguments that were found
						for arguments in curr_arguments_list:
							if set(tmp_pattern.keys()) == set(arguments.keys()):
								total_arguments.append(arguments)

	return total_arguments

def extract_args_from_nominal(nomlex_entries, sent="", dependency_tree=None,
							  limited_patterns_func=None, limited_indexes=None,
							  keep_arguments_locations=False, get_all_possibilities=False):
	"""
	Extracts the arguments of the nominalizations in the given sentence
	The given sentence can be presented using a string, or a dependency tree
	A sentence is preferable if both are given
	:param nomlex_entries: NOMLEX entries (a dictionary {nom: ...})
	:param sent: a sentence (string), optional
	:param dependency_tree: a dependency tree (list), optional
	:param limited_patterns_func: a limited function that should return a limited list of patterns according to the dependency_tree and the nom_index
	:param limited_indexes: a limited list indexes in which to search for nominalizations
	:param keep_arguments_locations: define whether to return also the locations of the arguments or not
	:param get_all_possibilities: define whether to return all the arguments possibilities or to choose the best ones
	:return: a dictionary of lists of dictionaries
			 dictionary of each founded nominalization (nom, original_nom, index) -> list of each suitable pattern -> dictionary of arguments
	"""

	if dependency_tree is None:
		dependency_tree = []

	# Not legal cases (the method must get a sentence or\and a dependency tree of a sentence)
	if sent == "" and dependency_tree == []:
		return {}

	# Checking what is the input base form that is given, and getting the dependency tree with it
	if sent != "" and dependency_tree == []:
		# Cleaning the given sentence
		while sent.startswith(" "):
			sent = sent[1:]

		while sent.endswith(" "):
			sent = sent[:-1]

		sent = sent.replace("’", "'").replace("\n", "").replace("\r\n", "").replace("\r", "")

		if sent == "":
			return {}

		dependency_tree = get_dependency(sent)

	# Replacing the first upper letter only if the word isn't a name of something (using NER from spacy)
	# The replacement will be in the tree, and not in the sentence
	if dependency_tree != [] and dependency_tree[0][-2] == "":
		dependency_tree[0] = (dependency_tree[0][0], dependency_tree[0][1].lower(), dependency_tree[0][2],
							  dependency_tree[0][3], dependency_tree[0][4], dependency_tree[0][5],
							  dependency_tree[0][6], dependency_tree[0][7], dependency_tree[0][8],
							  dependency_tree[0][9][0].lower() + dependency_tree[0][9][1:])

	# Finding all the nominalizations in the sentence
	noms = []

	if not limited_indexes:
		limited_indexes = range(len(dependency_tree))

	for i in limited_indexes:
		# Nominalization must be a noun
		if dependency_tree[i][4] == "NOUN":
			for nom in DictsAndTables.all_noms_backwards.get(dependency_tree[i][2], []):
				noms.append((nom, dependency_tree[i][1], i))

	# Moving over all the founded nominalizations
	nom_args = {}
	for nom, original_nom, nom_index in noms:
		# Getting the suitable nominalization entry
		nom_entry = nomlex_entries[nom]

		# Getting all the possible arguments for the specific nominalization
		if not limited_patterns_func:
			possible_args = get_arguments(dependency_tree, nom_entry, nom_index)
		else:
			limited_patterns = limited_patterns_func(dependency_tree, nom_index)
			possible_args = get_arguments(dependency_tree, nom_entry, nom_index, patterns=limited_patterns)

		best_args = possible_args

		# Cutting down the arguments lists that were found if needed
		if not get_all_possibilities:
			# Finding the maximum number of arguments that were extracted
			best_num_of_args = 0
			for args in possible_args:
				if len(args.keys()) > best_num_of_args:
					best_num_of_args = len(args.keys())

			# Add all the "best arguments" that were extracted (best = maximum number of arguments)
			best_args = []
			best_args_items = [] # List of list of the items that were extracted (for singularity)
			for args in possible_args:
				# Checking the number of arguments in args, and singularity
				if len(args.keys()) == best_num_of_args and args.items() not in best_args_items:
					if keep_arguments_locations:
						new_args = defaultdict(tuple)

						# Ignoring temp values for each argument
						for arg_name, (index, _, arg_value) in args.items():
							new_args[arg_name] = (index, arg_value)
					else:
						new_args = defaultdict(str)

						# Ignoring temp values for each argument
						for arg_name, (_, _, arg_value) in args.items():
							new_args[arg_name] = arg_value

					best_args.append(new_args)
					best_args_items.append(args.items())

		nom_args.update({(nom, original_nom, nom_index): best_args})

	return nom_args