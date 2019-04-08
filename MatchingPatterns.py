import DictsAndTables
from DictsAndTables import get_subentries_table
from VerbalPatterns import process_a_sentence
from NominalPatterns import extract_patterns_from_nominal

def clean_pattern(pattern):
	"""
	Cleans a pattern for not important arguments and data before the comparing the patterns
	:param pattern: a pattern of arguments (as a dict)
	:return: the cleaned pattern (also a dict)
	"""

	preps_subentries = [i for i, _, _ in get_subentries_table() if i.startswith("pval")] + ["wh"]

	# Removing the preposition word (in, on, about, ...) from preposition arguments
	for prep in preps_subentries:
		if prep in pattern.keys() and len(pattern[prep]):
			pattern[prep] = " ".join(pattern[prep].split(" ")[1:])

	# We want to match the lower case of the arguments
	for argument in pattern.keys():
		pattern[argument] = pattern[argument].lower()

	# Removing subcat argument
	pattern.pop("subcat")

	return pattern

def match_patterns(nomlex_entries, verbal_sentence, nominal_sentence):
	"""
	Finds all the exact matching between the arguments of the verb and nominalizations
	The verbal sentence defines that rule that needed to be found in the nominal sentences
	:param nomlex_entries: NOMLEX entries (a dictionary nom: ...)
	:param verbal_sentence: a simple sentence with a main verb
	:param nominal_sentence: a complex sentence that may include nominalizations
	:return: a list of the nominalizations in the nominal sentence that their arguments match the arguments of the main verb in the verbal sentences
			 and a status of the best nominalization match that was found
	"""

	DictsAndTables.should_print = False

	# Getting the arguments for the verb in the sentence (= processing the sentence)
	possible_verb_arguments = process_a_sentence(verbal_sentence)

	# Getting the arguments for all the nouns in the sentence (= processing the sentence)
	noms_arguments = extract_patterns_from_nominal(nomlex_entries, nominal_sentence)

	matching_noms = {}
	statuses = []
	max_exact_args = 0

	# There may be many possible arguments dictionaries
	for verb_arguments in possible_verb_arguments:
		temp_verb_arguments = clean_pattern(verb_arguments.copy())
		saved_verb_arguments = temp_verb_arguments.copy()

		# Checking each nominalization that was found
		for nom, possible_nom_arguments in noms_arguments.items():
			current_matching_patterns = []

			# And all its possible arguments that were extracted
			for nom_arguments in possible_nom_arguments:
				temp_nom_arguments = clean_pattern(nom_arguments.copy())

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
					statuses.append("exact match")
				elif temp_verb_arguments["verb"] == temp_nom_arguments["verb"]:
					if nom_arguments["subcat"] == verb_arguments["subcat"]:
						# More than subcat and verb
						num_of_exact_match_args = len([arg for arg in temp_verb_arguments.keys() if arg in temp_nom_arguments.keys() and temp_verb_arguments[arg] == temp_nom_arguments[arg]])
						if num_of_exact_match_args > 1:
							max_exact_args = num_of_exact_match_args
						else: # Only verb and subcat
							statuses.append("verb and subcat match")
					else: # Only verb
						statuses.append("verb match")

				temp_verb_arguments = saved_verb_arguments.copy()

			if current_matching_patterns != []:
				matching_noms.update({nom: (verb_arguments, current_matching_patterns)})

	# Finding the best matching that was found (this will be that total matching status)
	status = "not found any match"
	if "exact match" in statuses:
		status = "exact match"
	elif max_exact_args > 2:
		status = str(max_exact_args) + " arguments exact match"
	elif "verb and subcat match" in statuses:
		status = "verb and subcat match"
	elif "verb match" in statuses:
		status = "verb match"

	DictsAndTables.should_print = True

	return matching_noms, status