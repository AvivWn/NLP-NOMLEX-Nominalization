import json
import itertools
from collections import Counter
from allennlp.predictors.predictor import Predictor
from nltk.stem import PorterStemmer

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

	# Creating some patterns for the suitable case
	if objects_subentry != "NONE" and subjects_subentry != "NONE":
		objects = update_option(list(objects_subentry.keys()), subcat_info, "OBJECT")
		subjects = list(subjects_subentry.keys())

		if subjects == []:
			subjects = default_subjects
		else:
			subjects = update_option(subjects, subcat_info, "SUBJECT")

		if "SUBJECT" not in required_list:
			patterns += list(itertools.product(["NONE"], objects, ind_objects, pvals, pvals1, pvals2))

		if "OBJECT" not in required_list:
			patterns += list(itertools.product(subjects, ["NONE"], ind_objects, pvals, pvals1, pvals2))

		patterns += list(itertools.product(subjects, objects, ind_objects, pvals, pvals1, pvals2))
	elif objects_subentry != "NONE":
		objects = update_option(list(objects_subentry.keys()), subcat_info, "OBJECT")
		patterns += list(itertools.product(["NONE"], objects, ind_objects, pvals, pvals1, pvals2))
	elif subjects_subentry != "NONE":
		subjects = update_option(list(subjects_subentry.keys()), subcat_info, "SUBJECT")
		patterns += list(itertools.product(subjects, ["NONE"], ind_objects, pvals, pvals1, pvals2))

	patterns = list(set(patterns))

	# Deleting illegal patterns
	for pattern in patterns:
		p_subject, p_object, p_indobject, p_pval, p_pva1, p_pval2 = pattern
		if p_subject == 'NONE' and p_object == 'NONE' and p_indobject == 'NONE' and p_pval == 'NONE' and p_pva1 == 'NONE' and p_pval2 == 'NONE':
			patterns.remove(pattern)
		elif (p_subject == 'DET-POSS' and p_object == 'DET-POSS') or \
			 (p_subject == 'DET-POSS' and p_indobject == 'DET-POSS') or \
			 (p_object == 'DET-POSS' and p_indobject == 'DET-POSS'):
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


def process_phrase_tree(sent_phrase_tree, index):
	sub_phrase_trees = []

	while sent_phrase_tree[index] != ")":
		if sent_phrase_tree[index] == "(":
			index, sub_phrase_tree = process_phrase_tree(sent_phrase_tree, index + 1)
			sub_phrase_trees.append(sub_phrase_tree)
		else:
			sub_phrase_trees.append(sent_phrase_tree[index])
			index += 1

	if len(sub_phrase_trees) == 2:
		new_phrase_tree = {sub_phrase_trees[0]: [sub_phrase_trees[1]]}
	else:
		new_phrase_tree = {sub_phrase_trees[0]: sub_phrase_trees[1:]}

	return index + 1, new_phrase_tree

def get_phrase(phrase_tree):
	if type(phrase_tree) == str:
		return phrase_tree

	str_phrase = ""

	for _, sub_phrase_tree in phrase_tree.items():
		print(sub_phrase_tree)
		if type(sub_phrase_tree) == str:
			if str_phrase != "":
				str_phrase += " "

			str_phrase += sub_phrase_tree
		else:
			for sub_sub_phrase_tree in sub_phrase_tree:
				sub_sub_phrase = get_phrase(sub_sub_phrase_tree)

				if str_phrase != "":
					str_phrase += " "

				str_phrase += sub_sub_phrase

	return str_phrase

def search_phrase(phrase_tree, searched_tag):
	if type(phrase_tree) == str:
		return []

	wanted_phrases = []

	for phrase_tag, sub_phrase_tree in phrase_tree.items():
		if phrase_tag == searched_tag:
			wanted_phrases.append({phrase_tag: sub_phrase_tree})
		else:
			for sub_sub_phrase_tree in sub_phrase_tree:
				sub_wanted_phrases = search_phrase(sub_sub_phrase_tree, searched_tag)

				if sub_wanted_phrases != []:
					wanted_phrases += sub_wanted_phrases

	return wanted_phrases

def get_sub_phrases(phrase_tree, phrases_tags):
	index = 0
	phrases = []

	for sub_phrase_tree in phrase_tree:
		for tag, sub_phrase in sub_phrase_tree.items():
			if index < len(phrases_tags):
				if tag == phrases_tags[index]:
					phrases.append({tag: sub_phrase})
					index += 1
				else:
					phrases = []
					index = 0

	return phrases

def detect_comlex_subcat(sent, verb):
	predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
	phrase_tree = predictor.predict(sentence=sent)['trees']
	#print(phrases_tree)

	# Moving over each line in the input file
	# Spacing up all the opening\closing brackets
	temp_splitted_line = phrase_tree.replace("(", " ( ").replace(")", " ) ").replace(") \n", ")\n").replace("\"", "").split(' ')
	splitted_line = []

	for i in range(len(temp_splitted_line)):
		if temp_splitted_line[i] != '':
			splitted_line.append(temp_splitted_line[i].replace('\n', ''))

	new_sent = splitted_line

	_, phrase_tree = process_phrase_tree(new_sent, 1)

	print(phrase_tree)

	vp_phrases_trees = search_phrase(phrase_tree, "VP")[0]
	print(vp_phrases_trees)

	pval = "NONE"
	pval1 = "NONE"
	pval2 = "NONE"
	adverb = "NONE"
	subcat = "NONE"

	np_pp_phrases_trees = get_sub_phrases(vp_phrases_trees["VP"], ["NP", "PP"])
	pp_pp_phrases_trees = get_sub_phrases(vp_phrases_trees["VP"], ["PP", "PP"])
	np_np_phrases_trees = get_sub_phrases(vp_phrases_trees["VP"], ["NP", "NP"])
	np_phrases_trees = get_sub_phrases(vp_phrases_trees["VP"], ["NP"])

	if len(np_pp_phrases_trees) == 2:
		pp_phrases_tree = get_sub_phrases(np_pp_phrases_trees[0]["NP"], ["PP"])

		if len(pp_phrases_tree) == 1:
			pval1 = get_phrase(pp_phrases_tree[0])
			pval2 = get_phrase(np_pp_phrases_trees[1])
			subcat = "NOM-PP-PP"
		else:
			pval = get_phrase(np_pp_phrases_trees[1])
			print("print", np_pp_phrases_trees[1])
			subcat = "NOM-NP-PP"

	elif len(pp_pp_phrases_trees) == 2:
		pval1 = get_phrase(pp_pp_phrases_trees[0])
		pval2 = get_phrase(pp_pp_phrases_trees[1])
		subcat = "NOM-PP-PP"

	elif len(np_phrases_trees) == 1:
		subcat = "NOM-NP"

	else:
		subcat = "NONE"

	return {"subcat": subcat, "pval": pval, "pval1": pval1, "pval2": pval2, "adverb": adverb}

def process_a_sentence(sent):
	"""
	Processes a sentence
	:param sent: the sentence that was processed
	:return: the founded arguments of the verb in the sentence
	"""

	verb = "NONE"
	original_verb = "NONE"
	subject = "NONE"
	object = "NONE"
	indobject = "NONE"

	predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz")
	dependency_tree = predictor.predict(sentence=sent)

	# Assuming- Choosing the first verb
	verb_args = dependency_tree.get('verbs', [{}])[0].get('description', '').split("]")
	print(verb_args)

	for verb_arg in verb_args:
		if "ARG0:" in verb_arg:
			subject = verb_arg[verb_arg.index(":") + 2:]

		if "ARG1:" in verb_arg:
			object = verb_arg[verb_arg.index(":") + 2:]

		if "ARG2:" in verb_arg:
			indobject = verb_arg[verb_arg.index(":") + 2:]

		if "V:" in verb_arg:
			original_verb = verb_arg[verb_arg.index(":") + 2:]

			stemmer = PorterStemmer()
			verb = stemmer.stem(original_verb)

	subcat_arguments = detect_comlex_subcat(sent, original_verb)

	subject = remove_last_space(subject.replace(subcat_arguments["pval1"], ""))
	subject = remove_last_space(subject.replace(subcat_arguments["pval2"], ""))
	object = remove_last_space(object.replace(subcat_arguments["pval1"], ""))
	object = remove_last_space(object.replace(subcat_arguments["pval2"], ""))

	if indobject == subcat_arguments["pval1"] or indobject == subcat_arguments["pval2"]:
		indobject = "NONE"

	arguments = {"verb": verb, "subject": subject, "object": object, "indobject": indobject}

	arguments.update(subcat_arguments)

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

	pattern = {"subject": pattern[0], "object": pattern[1], "indobject": pattern[2], "pval": pattern[3], "pval1": pattern[4], "pval2": pattern[5]}

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

	if pre_nom == "":
		pre_nom = det + " "

	# Adding the nominalization
	sentence = pre_nom + nominalization

	# Getting all the prepositions the appeared in the pattern
	post_preps = []
	for role, role_type in pattern.items():
		if role_type.startswith("PP-"):
			post_preps.append([role_type.replace("PP-", "").lower(), arguments[role]])
		elif (role == "pval" or role == "pval1" or role == "pval2") and role_type != "NONE" and role_type != "none":
			post_preps.append([role_type.lower(), arguments[role].split(" ")[0]])

	# Finally, adding the relevant prepositions from the pattern (in any order)
	for preps_order in itertools.permutations(post_preps, len(post_preps)):
		temp_sentence = sentence
		for prep in preps_order:
			temp_sentence += " " + prep[0] + " " + prep[1]

		sentences.append(temp_sentence)

	for i in range(len(sentences)):
		sentences[i] = sentences[i].replace("the the ", "the ").replace("the a ", "a ").replace("the an ", "an ")
		sentences[i] = sentences[i].replace("she's", "her").replace("he's", "his").replace("i's", "my").replace("they's", "their").replace("we's", "our").replace("it's", "its")
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

	return nom_sentences



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







def remove_last_space(sent):
	if sent.endswith(" "):
		sent = sent[:-1]

	return sent




###################################################### Main ######################################################

def main(json_file_name, sent):
	json_data = load_json_data(json_file_name)

	all_patterns, _ = extract_nom_patterns(json_data)
	#print(all_patterns)
	#print(len(all_patterns))

	print(verbal_to_nominal(json_data, sent))

	"""
	other_sent = "(S (NP (NNP IBM)) (VP (VBD appointed) (NP (NNP Alice))))"

	# Moving over each line in the input file
	# Spacing up all the opening\closing brackets
	temp_splitted_line = other_sent.replace("(", " ( ").replace(")", " ) ").replace(") \n", ")\n").replace("\"", "").split(' ')
	splitted_line = []

	for i in range(len(temp_splitted_line)):
		if temp_splitted_line[i] != '':
			splitted_line.append(temp_splitted_line[i].replace('\n', ''))

	new_sent = [splitted_line]

	print(LispToJson.get_list(new_sent, 0, 1))
	"""

if __name__ == '__main__':
	"""
	Command line arguments-
		json_file_name sentence
	"""
	import sys

	main(sys.argv[1], sys.argv[2])