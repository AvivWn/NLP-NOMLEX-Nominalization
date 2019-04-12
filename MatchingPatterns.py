import DictsAndTables
from DictsAndTables import get_subentries_table
from VerbalPatterns import arguments_for_noms
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
	#pattern.pop("subcat")

	return pattern

def match_patterns(nomlex_entries, verbal_sentence, nominal_sentence):
	"""
	Finds all the exact matching between the arguments of the verb and nominalizations
	The verbal sentence defines the rule that needed to be found in the nominal sentences
	:param nomlex_entries: NOMLEX entries (a dictionary nom: ...)
	:param verbal_sentence: a simple sentence with a main verb
	:param nominal_sentence: a complex sentence that may include nominalizations
	:return: a list of the nominalizations in the nominal sentence that their arguments match the arguments of the main verb in the verbal sentences
	"""

	DictsAndTables.should_print = False

	# Getting the arguments for the verb in the sentence (= processing the sentence), according to the nominalizations
	verb_arguments_for_noms = arguments_for_noms(nomlex_entries, verbal_sentence)

	# Getting the arguments for all the nouns in the sentence (= processing the sentence)
	noms_arguments = extract_patterns_from_nominal(nomlex_entries, nominal_sentence)

	matching_noms = {}

	# Checking each nominalization that was found
	for nom, possible_nom_arguments in noms_arguments.items():
		# The relevant nominalization are only those that cam from the verb in the verbal sentence
		if nom[0] in verb_arguments_for_noms.keys():
			verb_arguments = clean_pattern(verb_arguments_for_noms[nom[0]].copy())
			current_matching_patterns = []

			# And all its possible arguments that were extracted
			for nom_arguments in possible_nom_arguments:
				cleaned_nom_arguments = clean_pattern(nom_arguments.copy())

				# Removing the subject if it is the nominalization
				if "subject" in cleaned_nom_arguments and "subject" in verb_arguments and cleaned_nom_arguments["subject"] == nom[0]:
					cleaned_nom_arguments.pop("subject")
					verb_arguments.pop("subject")

				# Removing the object if it is the nominalization
				elif "object" in cleaned_nom_arguments and "object" in verb_arguments and cleaned_nom_arguments["object"] == nom[0]:
					cleaned_nom_arguments.pop("object")
					verb_arguments.pop("object")

				# Comparing between the current pair of arguments
				if verb_arguments["verb"] == cleaned_nom_arguments["verb"]:
					if cleaned_nom_arguments["subcat"] == verb_arguments["subcat"]:
						# At least a matching of both the verb and the subcat
						current_matching_patterns.append(nom_arguments)

			if current_matching_patterns != []:
				if nom in matching_noms.keys():
					# Adding only the new pattern matches
					new_matching_patterns = []
					for found_pattern in current_matching_patterns:
						exist = False
						for match_pattern in matching_noms[nom]:
							if match_pattern == found_pattern:
								exist = True

						if not exist:
							new_matching_patterns.append(found_pattern)

					matching_noms.update({nom: matching_noms[nom] + new_matching_patterns})
				else:
					matching_noms.update({nom: current_matching_patterns})

	DictsAndTables.should_print = True

	return matching_noms