import json
import argparse
import os
import numpy as np
import pickle


############################################### Loading and Saving ###############################################

def load_json_data(json_file_name):
	"""
	Loads the data from a json file
	:param json_file_name: The name of the file that needed to be loaded
	:return: The the json data (basically a dictionary object)
	"""

	with open(json_file_name, 'r') as inputfile:
		data = json.load(inputfile)

	return data

def load_txt_file(txt_file_name):
	"""
	Loads the data from a txt file with the given name
	:param txt_file_name: The name of the file that needed to be loaded
	:return: The data in the file (as list of pairs of sentence and dependency tree)
	"""

	with open(txt_file_name, "r") as inputfile:
		data = inputfile.readlines()

	# Is this file is parsed file? Determine according to the first line
	if len(data[0].split("\t")) == 10:
		parsed_data = []
		curr_sent_data = []
		curr_sent = ""

		for line in data:
			if line == "\n":
				parsed_data.append((curr_sent, curr_sent_data))
				curr_sent_data = []
				curr_sent = ""

			else:
				line = line.replace("\n", "").replace("\r\n", "").replace("\r", "")
				index, word, lemma, tag, pos, head_id, dep, ent_iob, ent_type, str_sub_tree = line.split("\t")

				index = int(index)
				head_id = int(head_id)

				if curr_sent != "":
					curr_sent += " "

				curr_sent += word

				curr_sent_data.append((index, word, lemma, tag, pos, head_id, dep, ent_iob, ent_type, str_sub_tree))

		data = parsed_data

	return data



#################################################### Modules #####################################################

def verb_module(nomlex_entries, inputs):
	from VerbalPatterns import verbal_to_nominal

	num_of_checked_sentences = 0
	total_num_of_sentences = len(inputs)

	# Use a random order of the data
	random_indexes = np.arange(len(inputs))
	if DictsAndTables.shuffle_data: np.random.shuffle(random_indexes)

	for i in random_indexes:
		arranged_print(inputs[i])
		separate_line_print(verbal_to_nominal(nomlex_entries, inputs[i]))
		arranged_print("")

		num_of_checked_sentences += 1
		if not DictsAndTables.should_print_to_screen and total_num_of_sentences != 1:
			if num_of_checked_sentences == 1:
				print("Scanning " + str(num_of_checked_sentences) + "/" + str(total_num_of_sentences) + " sentences!")
			else:
				print("\033[1AScanning " + str(num_of_checked_sentences) + "/" + str(
					total_num_of_sentences) + " sentences!")

def nom_moudle(nomlex_entries, inputs):
	from NominalPatterns import extract_args_from_nominal
	from ExtractNomlexPatterns import extract_nom_patterns

	num_of_checked_sentences = 0
	total_num_of_sentences = len(inputs)

	# Use a random order of the data
	random_indexes = np.arange(len(inputs))
	if DictsAndTables.shuffle_data: np.random.shuffle(random_indexes)

	limited_patterns_func = None

	# In case of big amount of data, extract the patterns for each nominalization in the given lexicon before
	if total_num_of_sentences > 1:
		limited_noms_dict = extract_nom_patterns(nomlex_entries)
		limited_patterns_func = lambda dep_tree, nom_index, nom: limited_noms_dict.get(nom, ([],[]))[1]

	for i in random_indexes:
		if type(inputs[i]) == tuple:  # Input is already parsed
			sent, dep = inputs[i]
			noms_arguments_list = extract_args_from_nominal(nomlex_entries, sent=sent, dependency_tree=dep, limited_patterns_func=limited_patterns_func)
		else:  # Input is the sentence string
			sent = inputs[i]
			noms_arguments_list = extract_args_from_nominal(nomlex_entries, sent=sent, limited_patterns_func=limited_patterns_func)

		if DictsAndTables.should_print_as_dataset:
			print_as_dataset(sent, noms_arguments_list)
		else:
			arranged_print(sent)
			separate_line_print(noms_arguments_list)
			arranged_print("")

		num_of_checked_sentences += 1
		if not DictsAndTables.should_print_to_screen and total_num_of_sentences != 1:
			if num_of_checked_sentences == 1:
				print("Scanning " + str(num_of_checked_sentences) + "/" + str(total_num_of_sentences) + " sentences!")
			else:
				print("\033[1AScanning " + str(num_of_checked_sentences) + "/" + str(
					total_num_of_sentences) + " sentences!")

def match_module(nomlex_entries, verbal_inputs, nominal_inputs):
	from MatchingPatterns import match_patterns

	# Counting the founded subcats so far
	DictsAndTables.subcats_counts = Counter()

	if DictsAndTables.output_loc != sys.stdout:
		subcats = list(set([i[0] for i in comlex_table] + ["NOM-INTRANS", "NOM-INTRANS-RECIP"]))
		with open(DictsAndTables.output_loc.name, 'r') as read_output_file:
			output_file_lines = read_output_file.readlines()

		for subcat in subcats:
			DictsAndTables.subcats_counts[subcat] = 0

		for line in output_file_lines:
			for subcat in subcats:
				if "'" + subcat + "'" in line:
					DictsAndTables.subcats_counts[subcat] += 1

	match_patterns(nomlex_entries, verbal_inputs, nominal_inputs)


###################################################### Main ######################################################

def main(args):
	"""
	The main function
	:param args: the command line arguments (as argparse.namespace object)
	:return: None
	"""

	if args.verb and len(args.input) != 1:
		print('error: the inputs list should contain exactly one input in case of --verb module')
		return

	if args.nom and len(args.input) != 1:
		print('error: the inputs list should contain exactly one input in case of --nom module')
		return

	if args.match and len(args.input) != 2:
		print('error: the inputs list should contain exactly two inputs in case of --match module')
		return

	nomlex_entries = json.load(args.lexicon[0])
	DictsAndTables.all_noms, DictsAndTables.all_noms_backwards = get_all_of_noms(nomlex_entries)

	# Each input can be a string sentence or a name of file
	# Name of file is preferable
	inputs = []
	for input_str in args.input:
		if os.path.isfile(input_str):
			# Save the list in binary file for next time
			if not os.path.exists(input_str + "_as_list"):
				input_data = load_txt_file(input_str)

				with open(input_str + "_as_list", "wb") as patterns_file:
					pickle.dump(input_data, patterns_file)
			else:
				# Used the last saved file
				with open(input_str + "_as_list", "rb") as patterns_file:
					input_data = pickle.load(patterns_file)
		else:
			input_data = [input_str]
		inputs.append(input_data)

	DictsAndTables.output_loc = args.output[0]

	# Verb module
	if args.verb:
		verb_module(nomlex_entries, inputs[0])

	# Nominalization module
	if args.nom:
		nom_moudle(nomlex_entries, inputs[0])

	# Matching module
	if args.match:
		match_module(nomlex_entries, inputs[0], inputs[1])

	DictsAndTables.output_loc.close()

if __name__ == '__main__':
	import sys

	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
									 description='This program has tree main modules:\n'
												 '1. Extracting arguments from sentences with a main verb, and translating the sentence to nominal form\n'
												 '\tExample- "[A0 IBM] appointed [A1 Alice Smith]" --> arguments of "appoint"\n'
												 '\t\tNote: brackets and arguments names (limited to one word without spaces) are optional\n'
												 '2. Extracting arguments from sentences with nominalizations\n'
												 '\tExample- "Alice Smith\'s appointment by IBM" --> arguments of "appointment"\n'
												 '3. Finding matches between verbal sentences and nominal sentences\n'
												 '\tExample- "[A0 IBM] appointed [A1 Alice Smith]", "Alice Smith\'s appointment by IBM" --> match\n'
												 '\t\tNote: The order in important (verbal_sentences, nominals_sentences)\n\n'
												 'Currently the nominalizations are taken from the nomlex lexicon.')

	parser.add_argument('-lexicon', nargs=1, type=argparse.FileType('r'), required=True, help='name of a NOMLEX lexicon file')

	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--verb', action='store_true', help='activate verbal module (1)')
	group.add_argument('--nom', action='store_true', help='activate nominal module (2)')
	group.add_argument('--match', action='store_true', help='activate match module (3)')

	parser.add_argument('-input', nargs='*', required=True, help='list of inputs for the chosen module, where each input can be:\n'
																 '- string sentence\n'
																 '- name of file that contain sentences or Spacy parsed sentences (nominal case)')

	parser.add_argument('-output', nargs=1, type=argparse.FileType('a'), default=[sys.stdout], help='the program results output in that file')

	args = parser.parse_args(sys.argv[1:])

	import DictsAndTables
	from collections import Counter
	from DictsAndTables import comlex_table, \
							   separate_line_print, arranged_print, print_as_dataset, get_all_of_noms

	main(args)