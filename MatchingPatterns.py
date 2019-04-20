import sys
import os
import numpy as np
import DictsAndTables
from collections import defaultdict

from DictsAndTables import get_subentries_table, get_comlex_table, \
						   seperate_line_print, arranged_print
from VerbalPatterns import extract_args_from_verbal
from NominalPatterns import extract_args_from_nominal



############################################### Matching Patterns ###############################################

def clean_pattern(pattern):
	"""
	Cleans a pattern from not important arguments and data before comparing the patterns
	:param pattern: a pattern of arguments (as a dict)
	:return: the cleaned pattern (also a dict)
	"""

	if DictsAndTables.should_clean:
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

def replace_names(pattern, matching_names):
	"""
	Replaces the names of argument in the pattern, according to the given dictionary
	:param pattern: a dictionary of arguments {arg_str: arg_value}
	:param matching_names: a dictionary that match arg_strs to new arg_names
	:return: a new pattern after changing the argument names according to the matching dictionary
	"""

	new_pattern = defaultdict()

	for arg_str, arg_value in pattern.items():
		if arg_str in matching_names.keys():
			arg_str = matching_names[arg_str]

		new_pattern[arg_str] = arg_value

	return new_pattern

def match_patterns(nomlex_entries, verbal_sentences, nominal_sentences):
	"""
	Finds all the exact matching between the arguments of the verb and nominalizations
	The verbal sentence defines the rule that needed to be found in the nominal sentences
	:param nomlex_entries: NOMLEX entries (a dictionary nom: ...)
	:param verbal_sentences: a list of simple sentences with a main verb
	:param nominal_sentences: a list of complex sentences that may include nominalizations
	"""

	last_should_print_status = DictsAndTables.should_print
	DictsAndTables.should_print = False

	subcats = list(set([i[0] for i in get_comlex_table()]))

	verbs_arguments_for_noms = []
	for verbal_sentence in verbal_sentences:
		# Getting the arguments for the verb in the sentence (= processing the sentence)
		verbs_arguments_for_noms.append(extract_args_from_verbal(nomlex_entries, verbal_sentence))

	# Use a random order of the nominal sentences (if needed)
	random_indexes = np.arange(len(nominal_sentences))
	if DictsAndTables.shuffle_data: np.random.shuffle(random_indexes)

	found_match_count = 0
	num_of_checked_sentences = 0
	total_num_of_sentences = len(nominal_sentences) + 1
	last_num_printed_lines = 1
	subcats_counts_str = ""
	first_print = True

	# Moving over all the nominal sentences
	for nom_sent_index in random_indexes:
		nominal_sentence = nominal_sentences[nom_sent_index]

		# nominal_sentence_str is the string of the sentence
		if type(nominal_sentence) == tuple:  # Input is already parsed
			nominal_sentence_str, nominal_sentence_dep = nominal_sentence

			# Getting the arguments for all the nouns in the sentence (= processing the sentence) using the given dependency links
			noms_arguments = extract_args_from_nominal(nomlex_entries, dependency_tree=nominal_sentence_dep)
		else:  # Input is the sentence string
			nominal_sentence_str = nominal_sentence

			# Getting the arguments for all the nouns in the sentence (= processing the sentence)
			noms_arguments = extract_args_from_nominal(nomlex_entries, sent=nominal_sentence_str)

		curr_matching_noms = {}

		# For each nominal sentence, moving each nominalization that was found
		for nom, possible_nom_arguments in noms_arguments.items():

			# Trying each verbal sentence
			for verb_arguments_for_noms in verbs_arguments_for_noms:

				# The relevant nominalization are only those that came from the verb in the verbal sentence
				if nom[0] in verb_arguments_for_noms.keys():
					verb_arguments = clean_pattern(verb_arguments_for_noms[nom[0]][0].copy())
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
								current_matching_patterns.append(replace_names(nom_arguments, verb_arguments_for_noms[nom[0]][1]))

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

		DictsAndTables.should_print = last_should_print_status

		# Printing results and updating counts (for debuging)
		arranged_print("'" + nominal_sentence_str + "'")
		seperate_line_print(curr_matching_noms)
		arranged_print("")

		if curr_matching_noms != {}:
			found_match_count += 1

			if DictsAndTables.output_loc != sys.stdout:
				# Updating the suitable subcat
				for subcat in subcats:
					if "'" + subcat + "'" in str(curr_matching_noms):
						DictsAndTables.subcats_counts[subcat] += 1
		else:
			num_of_checked_sentences += 1

		# Debuging prints- status
		if DictsAndTables.output_loc != sys.stdout:
			if curr_matching_noms != {} or first_print:
				subcats_counts_str = ""

				_, columns = os.popen('stty size', 'r').read().split()
				num_printed_lines = 1
				curr_line = ""

				for subcat, count in DictsAndTables.subcats_counts.most_common():
					curr_str = str(subcat) + "=" + str(count)

					if len(curr_line + curr_str + " | ") > int(columns):
						num_printed_lines += 1
						subcats_counts_str += "".join([" " for _ in range(int(columns) - len(curr_line + curr_str + " | "))]) + "\n"
						curr_line = ""

					subcats_counts_str += curr_str + " | "
					curr_line += curr_str + " | "

				last_num_printed_lines = num_printed_lines + 1

			if not DictsAndTables.should_print_to_screen:
				if not first_print:
					print("\033[" + str(last_num_printed_lines) + "A" + subcats_counts_str + "\n" + "Found " + str(found_match_count) + " matches, from scanning " + str(num_of_checked_sentences) + "/" + str(total_num_of_sentences) + " sentences!")
				else:
					first_print = False
					print(subcats_counts_str + "\n" + "Found " + str(found_match_count) + " matches, from scanning " + str(num_of_checked_sentences) + "/" + str(total_num_of_sentences) + " sentences!")

		DictsAndTables.should_print = False

	DictsAndTables.should_print = last_should_print_status