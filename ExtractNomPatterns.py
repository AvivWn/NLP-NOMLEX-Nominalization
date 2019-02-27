import json
import itertools
from collections import Counter
from allennlp.predictors.predictor import Predictor
from nltk.stem import WordNetLemmatizer

det = "the"

############################################### Extracting Patterns ##############################################

def update_option(options, info, role):
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

def get_nom_subcat_patterns(entry, subcat):
	"""
	Creates and returns the patterns for a suitable nominalization entry and sub-categorization
	:param entry: the specific nominalization entry
	:param subcat: thes specific sub-categorization
	:return: a list of patterns (tuples- (subject, object, indobject)
	"""

	# Getting the default subject roles
	verb_subj_info = entry.get("VERB-SUBJ", {"NONE": {}})
	default_subjects = update_option(list(verb_subj_info.keys()), verb_subj_info, None)

	patterns = []

	# Getting the required list
	required_list = list(entry.get("REQUIRED", {}).keys())

	# Trying to get object, subjects, and indirect-objects sub-entries
	subcat_info = entry.get("VERB-SUBC", {}).get(subcat, {})
	objects_subentry = subcat_info.get("OBJECT", {"NONE": {}})
	subjects_subentry = subcat_info.get("SUBJECT", {})
	ind_objects_subentry = subcat_info.get("IND-OBJ", {"NONE": {}})

	ind_objects = update_option(list(ind_objects_subentry.keys()), subcat_info, "IND-OBJ")

	# Special subcat patterns
	pvals = subcat_info.get("PVAL", ["NONE"])
	pvals1 = subcat_info.get("PVAL1", ["NONE"])
	pvals2 = subcat_info.get("PVAL2", ["NONE"])

	adverb = ["NONE"]

	if subcat == "NOM-ADVP" or subcat == "NOM-ADVP-PP":
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
			patterns += list(itertools.product(["NONE"], objects, ind_objects, pvals, pvals1, pvals2, adverb)) \
					  + list(itertools.product(["NONE"], objects, ind_objects, pvals2, pvals1, pvals, adverb))

		if "OBJECT" not in required_list:
			patterns += list(itertools.product(subjects, ["NONE"], ind_objects, pvals, pvals1, pvals2, adverb)) \
					  + list(itertools.product(subjects, ["NONE"], ind_objects, pvals2, pvals1, pvals, adverb))

		patterns += list(itertools.product(subjects, objects, ind_objects, pvals, pvals1, pvals2, adverb)) \
				  + list(itertools.product(subjects, objects, ind_objects, pvals2, pvals1, pvals, adverb))

	elif objects_subentry != "NONE":
		objects = update_option(list(objects_subentry.keys()), subcat_info, "OBJECT")
		patterns += list(itertools.product(["NONE"], objects, ind_objects, pvals, pvals1, pvals2, adverb)) \
				  + list(itertools.product(["NONE"], objects, ind_objects, pvals2, pvals1, pvals, adverb))

	elif subjects_subentry != "NONE":
		subjects = update_option(list(subjects_subentry.keys()), subcat_info, "SUBJECT")
		patterns += list(itertools.product(subjects, ["NONE"], ind_objects, pvals, pvals1, pvals2, adverb)) \
				  + list(itertools.product(subjects, ["NONE"], ind_objects, pvals2, pvals1, pvals, adverb))

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

	return patterns

def get_nom_patterns(entry, subcat=None):
	"""
	Returns the possible object and subject pairs for the given entry
	:param entry: a dictionary info of a specific nominalization
	:param subcat: a sub-categorization type, optional argument.
		   If subcat is None, than the extraction won't be specific for a given subcat.
	:return: a list of all possible pairs for a specific nominalization entry (list of tuples)
	"""

	patterns = []

	if subcat != None:
		patterns += get_nom_subcat_patterns(entry, subcat)
	else:
		for subcat in entry.get("VERB-SUBC", {}).keys():
			patterns += get_nom_subcat_patterns(entry, subcat)

	return patterns

def extract_nom_patterns(entries, subcat=None):
	"""
	Extracts all the nominalization patterns from the json file with the given name
	:param entries: the json formatted data to extract from (entries)
	:param subcat: a sub-categorization type, optional argument.
		   If subcat is None, than the extraction won't be specific for a given subcat.
	:return: a counted nominalization patterns that can be found in the file
	"""

	patterns_list = []
	patterns_dict = {}

	for nominalization, entry in entries.items():
		patterns = get_nom_patterns(entry, subcat=subcat)
		patterns_dict.update({nominalization: patterns})
		patterns_list += patterns

	patterns_counter = Counter(patterns_list)

	return patterns_counter, patterns_dict




############################################### Verbal to Nominal ################################################

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

	predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
	phrase_tree = predictor.predict(sentence=sent)['trees']

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

	verb = "NONE"
	subj = "NONE"
	obj = "NONE"
	indobj = "NONE"
	subcat = "NONE"
	pval = "NONE"
	pval1 = "NONE"
	pval2 = "NONE"
	adverb = "NONE"

	# Use the first NP, VP pair that was found in the phrases tree
	np_vp_phrases_trees = get_sub_phrases(phrases_tree["S"], ["NP", "VP"])

	if len(np_vp_phrases_trees) == 2:
		np_phrase_tree = np_vp_phrases_trees[0]
		vp_phrase_tree = np_vp_phrases_trees[1]

		wordnet_lemmatizer = WordNetLemmatizer()
		verb = wordnet_lemmatizer.lemmatize(get_phrase(vp_phrase_tree).split(" ")[0], 'v')

		subj = get_phrase(np_phrase_tree)

		np_pp_pp_phrases_trees = get_sub_phrases(vp_phrase_tree["VP"], ["NP", "PP", "PP"])
		pp_pp_phrases_trees = get_sub_phrases(vp_phrase_tree["VP"], ["PP", "PP"])
		np_pp_phrases_trees = get_sub_phrases(vp_phrase_tree["VP"], ["NP", "PP"])
		np_np_phrases_trees = get_sub_phrases(vp_phrase_tree["VP"], ["NP", "NP"])
		advp_pp_phrases_tree = get_sub_phrases(vp_phrase_tree["VP"], ["ADVP", "PP"])
		pp_phrases_tree = get_sub_phrases(vp_phrase_tree["VP"], ["PP"])
		advp_phrases_tree = get_sub_phrases(vp_phrase_tree["VP"], ["ADVP"])
		np_phrases_tree = get_sub_phrases(vp_phrase_tree["VP"], ["NP"])

		# Checking which subcat is the most suitable one
		# Pay attention that the order of the checking is important
		if len(np_pp_pp_phrases_trees) == 3:
			obj = get_phrase(np_pp_pp_phrases_trees[0])
			pval = get_phrase(np_pp_pp_phrases_trees[1])
			pval2 = get_phrase(np_pp_pp_phrases_trees[2])
			subcat = "NOM-NP-PP-PP"

		elif len(pp_pp_phrases_trees) == 2:
			pval = get_phrase(pp_pp_phrases_trees[0])
			pval2 = get_phrase(pp_pp_phrases_trees[1])
			subcat = "NOM-PP-PP"

		elif len(np_pp_phrases_trees) == 2:
			pp_phrases_tree = get_sub_phrases(np_pp_phrases_trees[0]["NP"], ["PP"])

			if len(pp_phrases_tree) == 1:
				pval = get_phrase(pp_phrases_tree[0])
				pval2 = get_phrase(np_pp_phrases_trees[1])
				subcat = "NOM-PP-PP"
			else:
				obj = get_phrase(np_pp_phrases_trees[0])
				pval = get_phrase(np_pp_phrases_trees[1])
				subcat = "NOM-NP-PP"

		elif len(np_np_phrases_trees) == 2:
			indobj = get_phrase(np_np_phrases_trees[0])
			obj = get_phrase(np_np_phrases_trees[1])
			subcat = "NOM-NP-NP"

		elif len(advp_pp_phrases_tree) == 2:
			adverb = get_phrase(advp_pp_phrases_tree[0])
			pval = get_phrase(advp_pp_phrases_tree[1])
			subcat = "NOM-ADVP-PP"

		elif len(pp_phrases_tree) == 1:
			pval = get_phrase(pp_phrases_tree[0])
			subcat = "NOM-PP"

		elif len(advp_phrases_tree) == 1:
			adverb = get_phrase(advp_phrases_tree[0])
			subcat = "NOM-ADVP"

		elif len(np_phrases_tree) == 1:
			obj = get_phrase(np_phrases_tree[0])
			subcat = "NOM-NP"

		else:
			subcat = "NOM-INTRANS"

	return {"verb": verb, "subject": subj, "object": obj, "indobject": indobj, "subcat": subcat, "pval": pval, "pval1": pval1, "pval2": pval2, "adverb": adverb}


def process_a_sentence(sent):
	"""
	Processes a sentence, returns its relevant arguments
	:param sent: the sentence that was processed
	:return: the founded arguments of the verb in the sentence (as a dictionary)
	"""

	arguments = detect_comlex_subcat(sent)
	print(arguments)

	return arguments

def pattern_to_sent(nominalization, pattern, arguments):
	"""
	Translates a single pattern into a sentence\s, using context arguments
	:param nominalization: the nominalization word
	:param pattern: a pattern, that should be translated
	:param arguments: some context arguments, that helps in the translation
	:return: list of suitable nominal sentences for the given data
	"""

	found_relevant_in_pattern = False

	if pattern[0] != "NONE" or pattern[1] != "NONE" or pattern[2] != "NONE":
		found_relevant_in_pattern = True

	pattern = {"subject": pattern[0], "object": pattern[1], "indobject": pattern[2],
			   "pval": pattern[3], "pval1": pattern[4], "pval2": pattern[5], "adverb": pattern[6]}

	if (arguments["subject"] == "NONE" and pattern["subject"] != "NONE") or \
	   (arguments["object"] == "NONE" and pattern["object"] != "NONE") or \
	   (arguments["indobject"] == "NONE" and pattern["indobject"] != "NONE"):
		return []

	sentences = []

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

	# Current assumption- treating all the adverbs as eval-adv
	if pattern["adverb"] == "eval-adv" and arguments["adverb"] != "NONE":
		found_relevant_in_pattern = True
		if pre_nom == "":
			pre_nom = det + " "
		pre_nom += arguments["adverb"] + " "

	if pre_nom == "":
		pre_nom = det + " "

	# Adding the nominalization
	sentence = pre_nom + nominalization

	if pattern["adverb"] == "loc&dir" and arguments["adverb"] != "NONE":
		found_relevant_in_pattern = True
		sentence += " " + arguments["adverb"]

	# Getting all the prepositions the appeared in the pattern
	post_preps = []
	for role, role_type in pattern.items():
		if type(role_type) == str and role_type.startswith("PP-"):
			post_preps.append([role_type.replace("PP-", "").lower(), arguments[role]])
		elif (role == "pval" or role == "pval1" or role == "pval2") and role_type != "NONE" and role_type == arguments[role].split(" ")[0]:
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

	for i in range(len(sentences)):
		sentences[i] = " " + sentences[i] + " "

		# Double determiners
		sentences[i] = sentences[i].replace(" the the ", " the ").\
									replace(" the a ", " a ").\
									replace(" the an ", " an ")

		# Pronouns
		sentences[i] = sentences[i].replace(" she's ", " her ").\
									replace(" he's ", " his ").\
									replace(" I's ", " my ").\
									replace(" they's ", " their ").\
									replace(" we's ", " our ").\
									replace(" it's ", " its ").\
									replace(" you's ", " your ")

		sentences[i] = sentences[i].replace(" by she ", " by her ").\
									replace(" by he ", " by him ").\
									replace(" by I ", " by me ").\
									replace(" by they ", " by them ").\
									replace(" by we ", " by us ")

		sentences[i] = sentences[i].replace(" the she ", " her "). \
									replace(" the he ", " his "). \
									replace(" the I ", " my "). \
									replace(" the they ", " their "). \
									replace(" the we ", " our "). \
									replace(" the it ", " its "). \
									replace(" the you ", " your ")

		# Pronoun + determiner
		sentences[i] = sentences[i].replace(" the her ", " her ").\
									replace(" the his ", " his ").\
									replace(" the my ", " my ").\
									replace(" the their ", " their ").\
									replace(" the our ", " our ").\
									replace(" the its ", " its ").\
									replace(" the your ", " your ")

		sentences[i] = sentences[i][1:-1]
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
	_, nom_patterns = extract_nom_patterns(relevant_entries, arguments["subcat"])

	# Creating all the nominalization suitable sentences for the given sentence
	nom_sentences = []
	for nominalization, patterns in nom_patterns.items():
		for pattern in patterns:
			nom_sentences += pattern_to_sent(nominalization, pattern, arguments)

	return list(set(nom_sentences))




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




###################################################### Main ######################################################

def main(json_file_name, sent):
	json_data = load_json_data(json_file_name)

	all_patterns, _ = extract_nom_patterns(json_data)
	#print(all_patterns)
	#print(len(all_patterns))

	print(verbal_to_nominal(json_data, sent))

if __name__ == '__main__':
	"""
	Command line arguments-
		json_file_name sentence
	"""
	import sys

	main(sys.argv[1], sys.argv[2])