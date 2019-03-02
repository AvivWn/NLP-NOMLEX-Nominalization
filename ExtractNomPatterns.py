import json
import itertools
from allennlp.predictors.constituency_parser import ConstituencyParserPredictor
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from collections import defaultdict
import spacy
nlp = spacy.load('en_core_web_sm')

det = "the"




############################################### Extracting Patterns ##############################################

def update_option(options, info, role=None):
	"""
	Updates the options in the right way
	:param options: the current possible options (list)
	:param info: where to get alternatives (for PP)
	:param role: helps to know where to get alternatives (for PP)
	:return: the updated possible options
	"""

	if "NOM-IS-SUBJ" in options or "NOM-IS-OBJ" in options:
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
			options += ["IND-OBJ-" + s.upper() for s in list(other_info.values())[0]]

	if "PP" in options:
		options.remove("PP")

		if role:
			PP_info = info[role]["PP"]
		else:
			PP_info = info["PP"]

		if len(PP_info.get("PVAL", [])) > 0:
			options += ["PP-" + s.upper() for s in list(PP_info.get("PVAL", []))]

	return options

def get_options(a_tuple):
	subjs, objs, indobjs, pvals, pvals1, pvals2, adverbs = a_tuple

	all_tuples = list(itertools.product(subjs, objs, indobjs, pvals, pvals1, pvals2, adverbs))

	if pvals != ["NONE"] and pvals2 != ["NONE"]:
		all_tuples += list(itertools.product(subjs, objs, indobjs, pvals, pvals1, pvals2, adverbs))

	return all_tuples

def get_nom_subcat_patterns(entry, main_subentry, subcat):
	"""
	Creates and returns the patterns for a suitable nominalization entry and sub-categorization
	:param entry: the specific nominalization entry
	:param subcat: the specific sub-categorization
	:param main_subentry: the main entry to search for the given subcat.
						  If the main entry is None, than those entry is the default "VERB-SUBC" entry.
	:return: a list of patterns, each pattern is a dictionary (can be also tuple sometimes)
	"""

	# Getting the default subject roles
	verb_subj_info = entry.get("VERB-SUBJ", {"NONE": {}})
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

	objects_subentry = subcat_info.get("OBJECT", {"NONE": {}})
	subjects_subentry = subcat_info.get("SUBJECT", {})
	ind_objects_subentry = subcat_info.get("IND-OBJ", {"NONE": {}})

	ind_objects = update_option(list(ind_objects_subentry.keys()), subcat_info, "IND-OBJ")

	# Special subcat patterns
	pvals = subcat_info.get("PVAL", ["NONE"])
	pvals1 = subcat_info.get("PVAL1", ["NONE"])
	pvals2 = subcat_info.get("PVAL2", ["NONE"])

	adverb = ["NONE"]

	if subcat == "NOM-ADVP" or subcat == "NOM-ADVP-PP" or subcat == "NOM-NP-ADVP":
		adverb = ["loc&dir"]

	adverb = subcat_info.get("NOM-SUBC", {}).get("ADJP", adverb)
	if adverb == {}: adverb = ["eval-adv"]

	# Creating some patterns for the suitable case
	if objects_subentry != "NONE" and subjects_subentry != "NONE":
		objects = update_option(list(objects_subentry.keys()), subcat_info, "OBJECT")
		subjects = list(subjects_subentry.keys())

		if subjects == []:
			subjects = default_subjects
		else:
			subjects = update_option(subjects, subcat_info, "SUBJECT")

		if "SUBJECT" not in required_list:
			patterns += get_options((["NONE"], objects, ind_objects, pvals, pvals1, pvals2, adverb))

		if "OBJECT" not in required_list:
			patterns += get_options((subjects, ["NONE"], ind_objects, pvals, pvals1, pvals2, adverb))

		patterns += get_options((subjects, objects, ind_objects, pvals, pvals1, pvals2, adverb))

	elif objects_subentry != "NONE":
		objects = update_option(list(objects_subentry.keys()), subcat_info, "OBJECT")
		patterns += get_options((["NONE"], objects, ind_objects, pvals, pvals1, pvals2, adverb))

	elif subjects_subentry != "NONE":
		subjects = update_option(list(subjects_subentry.keys()), subcat_info, "SUBJECT")
		patterns += get_options((subjects, ["NONE"], ind_objects, pvals, pvals1, pvals2, adverb))

	patterns = list(set(patterns))

	# Deleting illegal patterns
	for pattern in patterns:
		subj, obj, indobj, pval, pva1, pval2, adverb = pattern
		if subj == 'NONE' and obj == 'NONE' and indobj == 'NONE' and pval == 'NONE' and pva1 == 'NONE' and pval2 == 'NONE' and adverb == 'NONE' and subcat != "NOM-INTRANS":
			patterns.remove(pattern)
		elif (subj == 'DET-POSS' and obj == 'DET-POSS') or \
			 (subj == 'DET-POSS' and indobj == 'DET-POSS') or \
			 (obj == 'DET-POSS' and indobj == 'DET-POSS'):
				patterns.remove(pattern)
		elif (subj == 'PP-OF' and (pval == 'of' or pva1 == 'of' or pval2 == 'of')) or \
			 (obj == 'PP-OF' and (pval == 'of' or pva1 == 'of' or pval2 == 'of')) or \
			 (indobj == 'PP-OF' and (pval == 'of' or pva1 == 'of' or pval2 == 'of')):
				patterns.remove(pattern)

	not_subentry = subcat_info.get("NOT", {})
	for and_entry, _ in not_subentry.items():
		not_patterns = get_nom_subcat_patterns(entry, not_subentry, and_entry)

		for not_pattern in not_patterns:
			if not_pattern in patterns:
				patterns.remove(not_pattern)

	dicts_patterns = []
	for pattern in patterns:
		subj, obj, indobj, pval, pval1, pval2, adverb = pattern

		dict_pattern = defaultdict(str)
		if subj != "NONE": dict_pattern["subject"] = subj
		if obj != "NONE": dict_pattern["object"] = obj
		if indobj != "NONE": dict_pattern["indobject"] = indobj
		if pval != "NONE": dict_pattern["pval"] = pval
		if pval1 != "NONE": dict_pattern["pval1"] = pval1
		if pval2 != "NONE": dict_pattern["pval2"] = pval2
		if adverb != "NONE": dict_pattern["adverb"] = adverb

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
		if entry["VERB"] == verb:
			relevant_entries.update({nom: entry})

	return relevant_entries


def get_comlex_db():
	"""
	Returns a static list of tuples, that represents the needed roles for each sub-categorization
	:return: a static list of tuples
	"""

	comlex_db = [("NOM-NP-PP-PP", ["object", "pval", "pval2"]),
				 ("NOM-PP-PP", ["pval", "pval2"]),
				 ("NOM-NP-PP", ["object", "pval"]),
				 ("NOM-NP-NP", ["indobject", "object"]),
				 ("NOM-ADVP-PP", ["adverb", "pval"]),
				 ("NOM-NP-ADVP", ["object", "adverb"]),
				 ("NOM-PP", ["pval"]),
				 ("NOM-ADVP", ["adverb"]),
				 ("NOM-NP", ["object"])]

	return comlex_db

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

	if type(phrases_tree) == str:
		return phrases_tree

	str_phrase = ""

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

def get_sub_phrases(phrases_tree, phrases_tags):
	"""
	This function tries to find the phrases tags in the given list as sub-phrases of the given phrases tree
	:param phrases_tree: a dictionary that represents the phrases tree to search in
	:param phrases_tags: a phrases tags list, we are looking to find
	:return: the list of sub-phrases that were found (the output is relevant only if its length is len(phrases_tags)
	"""

	index = 0
	phrases = []

	for sub_phrases_tree in phrases_tree:
		for tag, sub_sub_phrases_tree in sub_phrases_tree.items():
			if index < len(phrases_tags):
				if tag == phrases_tags[index]:
					phrases.append({tag: sub_sub_phrases_tree})
					index += 1
				else:
					phrases = []
					index = 0

	return phrases

def detect_comlex_subcat(sent):
	"""
	Detects the comblex sub-categorization of the given sentence
	:param sent: a sentence string
	:return: an arguments dictionary with values that are relevant to the founded subcat
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

	arguments = defaultdict(str)

	# Use the first NP, VP pair that was found in the phrases tree
	np_vp_phrases_trees = get_sub_phrases(phrases_tree["S"], ["NP", "VP"])

	if len(np_vp_phrases_trees) == 2:
		np_phrase_tree = np_vp_phrases_trees[0]
		vp_phrase_tree = np_vp_phrases_trees[1]

		wordnet_lemmatizer = WordNetLemmatizer()
		arguments["verb"] = wordnet_lemmatizer.lemmatize(get_phrase(vp_phrase_tree).split(" ")[0], 'v')

		arguments["subject"] = get_phrase(np_phrase_tree)

		complex_db = get_comlex_db()

		found_subcat = False
		for subcat, suitable_roles in complex_db:
			if not found_subcat:
				tags_phrases = subcat.replace("NOM-", "").split("-")
				sub_phrases = get_sub_phrases(vp_phrase_tree["VP"], tags_phrases)

				if len(sub_phrases) == len(tags_phrases):
					found_subcat = True
					arguments["subcat"] = subcat

					for i in range(len(tags_phrases)):
						arguments[suitable_roles[i]] = get_phrase(sub_phrases[i])

		if not found_subcat:
			arguments["subcat"] = "NOM-INTRANS"

	return arguments


def process_a_sentence(sent):
	"""
	Processes a sentence, returns its relevant arguments
	:param sent: the sentence that was processed
	:return: the founded arguments of the verb in the sentence (as a dictionary)
	"""

	arguments = detect_comlex_subcat(sent)
	print(arguments)

	return arguments

def build_pre_nom(pattern, arguments):
	"""
	Builds the pre nominalization sentence
	:param pattern: a dictionary of roles and their types
	:param arguments: an arguments dictionary, contains some arguments of a certain nominalization
	:return: the string before the nominalization (by default it must be in the next order- subject > indobject > object)
	"""

	found_relevant_in_pattern = False

	if pattern["subject"] != "" or pattern["object"] != "" or pattern["indobject"] != "":
		found_relevant_in_pattern = True

	if (arguments["subject"] == "" and pattern["subject"] != "") or \
			(arguments["object"] == "" and pattern["object"] != "") or \
			(arguments["indobject"] == "" and pattern["indobject"] != ""):
		return []

	pre_nom = ""

	if pattern["subject"] == "DET-POSS":
		pre_nom += arguments["subject"] + "'s "
	elif pattern["subject"] == "N-N-MOD":
		pre_nom += det + " " + arguments["subject"] + " "

	if pattern["indobject"] == "DET-POSS":
		pre_nom += arguments["indobject"] + "'s "
	elif pattern["indobject"] == "N-N-MOD":
		if pre_nom == "":
			pre_nom += det + " " + arguments["indobject"] + " "
		else:
			pre_nom += arguments["indobject"] + " "

	if pattern["object"] == "DET-POSS":
		pre_nom += arguments["object"] + "'s "
	elif pattern["object"] == "N-N-MOD":
		if pre_nom == "":
			pre_nom += det + " " + arguments["object"] + " "
		else:
			pre_nom += arguments["object"] + " "

	# Adding the adverb as adj if it is eval-adv
	if pattern["adverb"] == "eval-adv" and arguments["adverb"] != "":
		found_relevant_in_pattern = True
		if pre_nom == "":
			pre_nom = det + " "

		adj = get_adj(arguments["adverb"])

		if adj:
			pre_nom += adj + " "

	if pre_nom == "":
		pre_nom = det + " "

	return pre_nom, found_relevant_in_pattern

def clean_sentence(sentence):
	"""
	Cleans the sentence from mistakes with pronouns and more
	:param sentence: a word sentnece
	:return: the cleaned sentence
	"""

	pronoun_dict = get_pronoun_dict()
	sent = sentence

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
		sent = sent.replace(" " + pronoun + "'s ", " " + forms_list[0] + " ").\
					replace(" " + pronoun[0].upper() + pronoun[1:] + "'s ", " " + forms_list[0] + " ")

		sent = sent.replace(" by " + pronoun + " ", " by " + forms_list[1] + " ").\
					replace(" by " + pronoun[0].upper() + pronoun[1:] + " ", " by " + forms_list[1] + " ")

		# Pronoun + determiner
		sent = sent.replace(" " + det + " " + pronoun + " ", " " + forms_list[0] + " ").\
					replace(" " + det + " " + pronoun[0].upper() + pronoun[1:] + " ", " " + forms_list[0] + " ")

		sent = sent.replace(" " + det + " " + forms_list[0] + " ", " " + forms_list[0] + " ").\
					replace(" " + det + " " + forms_list[0].upper() + forms_list[0][1:] + " ", " " + forms_list[0] + " ")

	return sent

def pattern_to_sent(nominalization, pattern, arguments):
	"""
	Translates a single pattern into a sentence\s, using context arguments
	:param nominalization: the nominalization word
	:param pattern: a pattern, that should be translated
	:param arguments: some context arguments, that helps in the translation
	:return: list of suitable nominal sentences for the given data
	"""

	sentences = []

	# Getting the sentence string before the nominalization
	pre_nom, found_relevant_in_pattern = build_pre_nom(pattern, arguments)

	# Adding the nominalization
	sentence = pre_nom + nominalization

	# Adding the adverb as regular adverb if it is loc&dir
	if pattern["adverb"] == "loc&dir" and arguments["adverb"] != "":
		found_relevant_in_pattern = True
		sentence += " " + arguments["adverb"]

	# Getting all the prepositions the appeared in the pattern
	post_preps = []
	for role, role_type in pattern.items():
		if type(role_type) == str and role_type.startswith("PP-"):
			post_preps.append([role_type.replace("PP-", "").lower(), arguments[role]])
		elif (role == "pval" or role == "pval1" or role == "pval2") and role_type != "" and role_type == arguments[role].split(" ")[0]:
			found_relevant_in_pattern = True
			post_preps.append([role_type.lower(), " ".join(arguments[role].split(" ")[1:])])

	if not found_relevant_in_pattern and arguments["subcat"] != "NOM-INTRANS":
		return []

	# Finally, adding the relevant prepositions from the pattern (in any order)
	for preps_order in itertools.permutations(post_preps, len(post_preps)):
		temp_sentence = sentence
		for prep in preps_order:
			temp_sentence += " " + prep[0] + " " + prep[1]

		sentences.append(temp_sentence)

	# Cleaning the resulted sentences
	for i in range(len(sentences)):
		sentences[i] = " " + sentences[i] + " "
		sentences[i] = clean_sentence(sentences[i])
		sentences[i] = sentences[i][1:-1] # Without the added spaces
		sentences[i] = sentences[i][0].upper() + sentences[i][1:]

	return sentences

def verbal_to_nominal(json_data, sent):
	"""
	Translates a verbal sentence into a nominal sentence, using nominalizations
	Assumption- the sentence contain only one verb
	:param json_data: the json formatted data
	:param sent: a given verbal sentence
	:return: a list of nominal suitable sentences for the given sentence
	"""

	# Getting the arguments for the verb in the sentence (= processing the sentence)
	arguments = process_a_sentence(sent)

	# Getting the relevant nominalization entries according to the verb that we found
	relevant_entries = get_nomlex_entries(json_data, arguments["verb"])

	# Extracting all the suitable nominalization patterns
	nom_patterns = extract_nom_patterns(relevant_entries, arguments["subcat"])

	# Creating all the nominalization suitable sentences for the given sentence
	nom_sentences = []
	for nominalization, patterns in nom_patterns.items():
		for pattern in patterns:
			nom_sentences += pattern_to_sent(nominalization, pattern, arguments)

	return list(set(nom_sentences))





############################################## Extracting Arguments ##############################################

def get_depedency(sent):
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

def pattern_to_UD(pattern):
	"""
	Translates a pattern into universal dependency sequence
	:param pattern: a given pattern (dictionary role: role_type)
	:return: a suitable sequence of universal dependency links (all the links are in outside direction)
	"""

	pattern_UD = defaultdict(list)

	for role, role_type in pattern.items():
		if role == "pval" or role == "pval1" or role == "pval2":
			pattern_UD[role] = ["prep_" + role_type.lower(), "pobj"]

		elif role == "adverb":
			if role_type == "eval-adv":
				pattern_UD[role] = ["amod"]
			elif role_type == "loc&dir":
				pattern_UD[role] = ["advmod"]

		elif role != "subcat":
			if role_type == "DET-POSS":
				pattern_UD[role] = ["poss"]
			elif role_type == "N-N-MOD":
				pattern_UD[role] = ["compound"]
			elif role_type.startswith("PP-"):
				pattern_UD[role] = ["prep_" + role_type.replace("PP-", "").lower(), "pobj"]

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

	# Deleting the ending " 's" in case that the role_type was DET-POSS
	if arg.endswith(" 's"):
		arg = arg[:-3]

	return arg

def extract_argument(dep_tree, dep_links, dep_curr_index):
	"""
	A recursive function that finds an argument using acording to the given dependency links
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
			arg = clean_argument(dep_tree[dep_curr_index][9])
			return [(dep_tree[dep_curr_index][0], arg)]

	if dep_curr_index == -1:
		return []

	arguments = []
	for i in range(len(dep_tree)):
		# Checking if the node link to the current node
		if dep_tree[i][5] - 1 == dep_curr_index:
			# Checking that the link type is right
			if dep_links[0].startswith("prep_"):
				splitted = dep_links[0].split("_")
				if dep_tree[i][6] == splitted[0] and dep_tree[i][2] == splitted[1]:
					arguments += extract_argument(dep_tree, dep_links[1:], i)

			elif dep_tree[i][6] == dep_links[0]:
				arguments += extract_argument(dep_tree, dep_links[1:], i)

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

	# Moving over all the possible patterns for the given nominalization
	# Trying to extract all the possible arguments for that nominalization
	for pattern in patterns:
		# Translating the pattern into universal dependencies sequence
		pattern_UD = pattern_to_UD(pattern)

		# Initiate the current arguments dictionary
		curr_arguments = defaultdict(tuple)
		curr_arguments["verb"] = (-1, nom_entry["VERB"])

		# Is the nominalization itself has a role in the sentence
		if "SUBJECT" in nom_entry["NOM-TYPE"].keys():
			curr_arguments["subject"] = (-1, dependency_tree[nom_index][1])
		elif "OBJECT" in nom_entry["NOM-TYPE"].keys():
			curr_arguments["object"] = (-1, dependency_tree[nom_index][1])

		curr_arguments_list = [curr_arguments]
		new_curr_arguements_list = curr_arguments_list.copy()

		# Looking for each argument (the order is important, because subject > indobject > object and not otherwise)
		for role in ["subject", "indobject", "object", "pval", "pval1", "pval2", "adverb"]:
			role_type = pattern_UD[role]

			if role_type != []:
				possible_arguments = extract_argument(dependency_tree, role_type, nom_index)

				# Checking all the possible arguments that were extracted for the current role
				if possible_arguments != []:
					for arguments in curr_arguments_list:
						for index, arg in possible_arguments:
							temp_arguments = arguments.copy()

							# Translate adjective to adverb if needed
							if role == "adverb" and pattern[role] == "eval-adv":
								arg = get_adv(arg)

							curr_indexes = [i for i, _ in temp_arguments.values()]
							if index not in curr_indexes:
								if role in ["subject", "indobject", "object"]:
									if pattern[role].startswith("PP-"):
										temp_arguments[role] = (-1, arg)
									elif index > max(curr_indexes):
										temp_arguments[role] = (index, arg)
								else:
									temp_arguments[role] = (index, arg)

							new_curr_arguements_list.append(temp_arguments)

					curr_arguments_list = new_curr_arguements_list.copy()

		total_arguments += new_curr_arguements_list.copy()

	return total_arguments

def extract_arguments(nomlex_entries, sent):
	"""
	Extracts the arguments of the nominalizations in the given sentence
	:param nomlex_entries: NOMLEX entries (a dictionary nom: ...)
	:param sent: a given sentence (string)
	:return: a dictionary of lists of dictionaries
			 dictionary of each founded nominalization (nom, index) -> list of each suitable pattern -> dictionary of arguments
	"""

	# Getting the dependency tree of the sentence
	dependency_tree = get_depedency(sent)

	# Finding all the nominalizations in the tree
	noms = []
	for i in range(len(dependency_tree)):
		if dependency_tree[i][2] in nomlex_entries.keys():
			noms.append((dependency_tree[i][2], i))

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
				for role, (_, arg) in args.items():
					new_args[role] = arg

				best_args.append(new_args)
				best_args_items.append(args.items())

		nom_args.update({(nom, nom_index): best_args})

	return nom_args




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

def get_best_word(word, possible_list):
	"""
	Returns the most relevant word in the possible list to the given word
	The most relevant is a word that starts the same as the given word
	:param word: a word
	:param possible_list: a list of words
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
		if i > best_subword_length:
			best_subword_length = i
			best_word = possible_word

	return best_word

def get_adj(word):
	"""
	Returns the best adjective that relates to the given word (if no adjective was found, None is returned)
	:param word: a word
	:return: an adjective that is most relevant to the given word, or None
	"""

	possible_adj = []
	for ss in wn.synsets(word):
		for lemmas in ss.lemmas():  # all possible lemmas
			for ps in lemmas.pertainyms():  # all possible pertainyms (the adjectives of a noun)
				possible_adj.append(ps.name())

	best_adj = get_best_word(word, possible_adj)

	return best_adj

def get_adv(word):
	"""
	Returns the best adverb that relates to the given word (if no adverb was found, None is returned)
	:param word: a word
	:return: an adverb that is most relevant to the given word, or None
	"""

	possible_adv = []
	for synset in list(wn.all_synsets('r')):
		if get_adj(synset.lemmas()[0].name()) == word:
			possible_adv.append(synset.lemmas()[0].name())

	best_adv = get_best_word(word, possible_adv)

	return best_adv


def get_pronoun_dict():
	pronoun_dict = {"he":["his", "him"], "she":["her", "her"], "it":["its", "its"], "they":["their", "them"], "we":["our", "us"], "i":["my", "me"]}

	return pronoun_dict




###################################################### Main ######################################################

def main(arguments):
	"""
	The main function
	:param arguments: the command line arguments
	:return: None
	"""

	if arguments[0] == "-patterns" and len(arguments) == 3:
		json_file_name = arguments[1]
		sent = arguments[2]

		json_data = load_json_data(json_file_name)
		print(verbal_to_nominal(json_data, sent))
	elif arguments[0] == "-args" and len(arguments) == 3:
		json_file_name = arguments[1]
		sent = arguments[2]

		json_data = load_json_data(json_file_name)
		print(extract_arguments(json_data, sent))

if __name__ == '__main__':
	"""
	Command line arguments-
		 -patterns json_file_name sentence
		 -args json_file_name sentence
	"""
	import sys

	main(sys.argv[1:])