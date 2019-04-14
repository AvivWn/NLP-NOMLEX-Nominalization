import sys
import numpy as np
import DictsAndTables
from DictsAndTables import get_subentries_table
from VerbalPatterns import arguments_for_noms
from NominalPatterns import extract_patterns_from_nominal

def clean_sentence(sent):
	"""
		Cleans a given sentence from starting with space or ending with new line sign
		:param sent: a sentece (string)
		:return: the cleaned sentence (as string)
	"""

	while sent.startswith(" "):
		sent = sent[1:]

	sent = sent.replace("\n", "").replace("\r\n", "").replace("\r", "")

	return sent

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

def match_patterns(nomlex_entries, verbal_sentences, nominal_sentences):
	"""
	Finds all the exact matching between the arguments of the verb and nominalizations
	The verbal sentence defines the rule that needed to be found in the nominal sentences
	:param nomlex_entries: NOMLEX entries (a dictionary nom: ...)
	:param verbal_sentences: a list of simple sentences with a main verb
	:param nominal_sentences: a list of complex sentences that may include nominalizations
	"""

	DictsAndTables.should_print = False

	verbs_arguments_for_noms = []
	for verbal_sentence in verbal_sentences:
		verbal_sentence = clean_sentence(verbal_sentence)

		# Getting the arguments for the verb in the sentence (= processing the sentence)
		verbs_arguments_for_noms.append(arguments_for_noms(nomlex_entries, verbal_sentence))

	# Use a random order of the nominal sentences
	random_indexes = np.arange(len(nominal_sentences))
	np.random.shuffle(random_indexes)

	found_match_count = 0
	num_of_checked_sentences = 0
	total_num_of_sentences = len(nominal_sentences)

	# Moving over all the nominal sentences
	for nom_sent_index in random_indexes:
		nominal_sentence = clean_sentence(nominal_sentences[nom_sent_index])

		# Getting the arguments for all the nouns in the sentence (= processing the sentence)
		noms_arguments = extract_patterns_from_nominal(nomlex_entries, nominal_sentence)

		curr_matching_noms = {}

		# For each nominal sentence, try each verbal sentence
		for i in range(len(verbs_arguments_for_noms)):
			# Checking each nominalization that was found
			for nom, possible_nom_arguments in noms_arguments.items():
				# The relevant nominalization are only those that cam from the verb in the verbal sentence
				if nom[0] in verbs_arguments_for_noms[i].keys():
					verb_arguments = clean_pattern(verbs_arguments_for_noms[i][nom[0]].copy())
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
						if nom in curr_matching_noms.keys():
							# Adding only the new pattern matches
							new_matching_patterns = []
							for found_pattern in current_matching_patterns:
								exist = False
								for match_pattern in curr_matching_noms[nom]:
									if match_pattern == found_pattern:
										exist = True

								if not exist:
									new_matching_patterns.append(found_pattern)

							curr_matching_noms.update({nom: curr_matching_noms[nom] + new_matching_patterns})
						else:
							curr_matching_noms.update({nom: current_matching_patterns})

		DictsAndTables.should_print = True

		if curr_matching_noms != {}:
			if DictsAndTables.should_print: print("'" + nominal_sentence + "'", file=DictsAndTables.output_loc)
			DictsAndTables.seperate_line_print(curr_matching_noms)
			print("", file=DictsAndTables.output_loc)
			DictsAndTables.output_loc.flush()
			found_match_count += 1
		else:
			num_of_checked_sentences += 1

		if DictsAndTables.output_loc != sys.stdout:
			print("\033[1AFound " + str(found_match_count) + " matches, from scanning " + str(num_of_checked_sentences) + "/" + str(total_num_of_sentences) + " sentences!")

		DictsAndTables.should_print = False

	DictsAndTables.should_print = True