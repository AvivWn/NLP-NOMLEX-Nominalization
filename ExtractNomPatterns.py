import json
import itertools
from collections import Counter
from allennlp.predictors.predictor import Predictor
from nltk.stem import PorterStemmer

import LispToJson

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

	# Creating some patterns for the suitable case
	if objects_subentry != "NONE" and subjects_subentry != "NONE":
		objects = update_option(list(objects_subentry.keys()), subcat_info, "OBJECT")
		subjects = list(subjects_subentry.keys())

		if subjects == []:
			subjects = default_subjects
		else:
			subjects = update_option(subjects, subcat_info, "SUBJECT")

		if "SUBJECT" not in required_list:
			patterns += list(itertools.product(["NONE"], objects, ind_objects))

		if "OBJECT" not in required_list:
			patterns += list(itertools.product(subjects, ["NONE"], ind_objects))

		patterns += list(itertools.product(subjects, objects, ind_objects))
	elif objects_subentry != "NONE":
		objects = update_option(list(objects_subentry.keys()), subcat_info, "OBJECT")
		patterns += list(itertools.product(["NONE"], objects, ind_objects))
	elif subjects_subentry != "NONE":
		subjects = update_option(list(subjects_subentry.keys()), subcat_info, "SUBJECT")
		patterns += list(itertools.product(subjects, ["NONE"], ind_objects))

	patterns = list(set(patterns))
	if ('NONE', 'NONE', 'NONE') in patterns:
		patterns.remove(('NONE', 'NONE', 'NONE'))

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

def detect_comlex_subcat(sent, verb):
	predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
	phrases_tree = predictor.predict(sentence=sent)['trees']
	#print(phrases_tree)

	num_of_brackets = 0
	curr_str = ""
	tags = []
	found_VP = False
	found_verb = False

	for char in phrases_tree:
		if char == "(":
			num_of_brackets += 1
		elif char == ")":
			num_of_brackets -= 1

			if found_VP and num_of_brackets == 1:
				break
		elif char == " ":
			if curr_str == "VP":
				found_VP = True

			if found_VP and curr_str == verb:
				found_verb = True

			if found_verb and (curr_str == "NP" or curr_str == "PP"):
				tags.append(curr_str)

			curr_str = ""
		else:
			curr_str += char

	print(tags)

	if " ".join(tags).count("PP") >= 2:
		return "NOM-PP-PP"
	elif "NP NP" in " ".join(tags):
		return "NOM-NP-NP"
	elif "NP PP" in " ".join(tags):
		return "NOM-NP-PP"
	elif "NP" in " ".join(tags):
		return "NOM-NP"

	return "NONE"

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

	arguments = {"verb": verb, "subject": subject, "object": object, "indobject": indobject}

	subcat = detect_comlex_subcat(sent, original_verb)
	arguments.update({"subcat": subcat})

	#print(arguments)
	return arguments

def pattern_to_sent(nominalization, pattern, arguments):
	"""
	Translates a single pattern into a sentence\s, using context arguments
	:param nominalization: the nominalization word
	:param pattern: a pattern, that should be translated
	:param arguments: some context arguments, that helps in the translation
	:return: list of suitable nominal sentences for the given data
	"""

	pattern = {"subject": pattern[0], "object": pattern[1], "indobject": pattern[2]}

	if (arguments["subject"] == "NONE" and pattern["subject"] != "NONE") or \
	   (arguments["object"] == "NONE" and pattern["object"] != "NONE") or \
	   (arguments["indobject"] == "NONE" and pattern["indobject"] != "NONE"):
		return []

	sentences = []

	# Building the sentence before the nominalization
	# The order of the words before the nominalization must be: subject > indobject > (direct) object
	if pattern["subject"] == "DET-POSS":
		if pattern["object"] == "N-N-MOD":
			if pattern["indobject"] == "N-N-MOD":
				pre_nom = arguments["subject"] + "'s " + arguments["indobject"] + " " + arguments["object"] + " "
			else:
				pre_nom = arguments["subject"] + "'s " + arguments["object"] + " "
		else:
			if pattern["indobject"] == "N-N-MOD":
				pre_nom = arguments["subject"] + "'s " + arguments["indobject"] + " "
			else:
				pre_nom = arguments["subject"] + "'s "

	elif pattern["subject"] == "N-N-MOD":
		if pattern["object"] == "N-N-MOD":
			if pattern["indobject"] == "N-N-MOD":
				pre_nom = det + " " + arguments["subject"] + " " + arguments["indobject"] + " " + arguments["object"] + " "
			elif pattern["indobject"] == "DET-POSS":
				pre_nom = det + " " + arguments["subject"] + " " + arguments["indobject"] + "'s " + arguments["object"] + " "
			else:
				pre_nom = det + " " + arguments["subject"] + " " + arguments["object"] + " "
		elif pattern["object"] == "DET-POSS":
			if pattern["indobject"] == "N-N-MOD":
				pre_nom = det + " " + arguments["subject"] + " " + arguments["indobject"] + " " + arguments["object"] + "'s "
			else:
				pre_nom = det + " " + arguments["subject"] + " " + arguments["object"] + "'s "
		else:
			if pattern["indobject"] == "N-N-MOD":
				pre_nom = det + " " + arguments["subject"] + " " + arguments["indobject"] + " "
			elif pattern["indobject"] == "DET-POSS":
				pre_nom = det + " " + arguments["subject"] + " " + arguments["indobject"] + "'s "
			else:
				pre_nom = det + " " + arguments["subject"] + " "

	else:
		if pattern["object"] == "N-N-MOD":
			if pattern["indobject"] == "N-N-MOD":
				pre_nom = det + " " + arguments["indobject"] + " " + arguments["object"] + " "
			elif pattern["indobject"] == "DET-POSS":
				pre_nom = arguments["indobject"] + "'s " + arguments["object"] + " "
			else:
				pre_nom = det + " " + arguments["object"] + " "
		elif pattern["object"] == "DET-POSS":
			if pattern["indobject"] == "N-N-MOD":
				pre_nom = det + " " + arguments["indobject"] + " " + arguments["object"] + "'s "
			else:
				pre_nom = arguments["object"] + "'s "
		else:
			if pattern["indobject"] == "N-N-MOD":
				pre_nom = det + " " + arguments["indobject"] + " "
			elif pattern["indobject"] == "DET-POSS":
				pre_nom = arguments["indobject"] + "'s "
			else:
				pre_nom = det + " "

	# Adding the nominalization
	sentence = pre_nom + nominalization

	# Getting all the prepositions the appeared in the pattern
	post_preps = []
	for role, role_type in pattern.items():
		if role_type.startswith("PP-"):
			post_preps.append([role_type.replace("PP-", "").lower(), arguments[role]])

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