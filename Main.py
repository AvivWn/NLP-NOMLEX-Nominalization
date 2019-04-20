import json
import argparse
import os
import numpy as np

import DictsAndTables
from collections import Counter
from DictsAndTables import get_comlex_table, \
						   seperate_line_print, arranged_print
from VerbalPatterns import verbal_to_nominal
from NominalPatterns import extract_args_from_nominal
from MatchingPatterns import match_patterns



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
	:return: The the file data (as list of lines)
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



###################################################### Main ######################################################

def main(args):
	"""
	The main function
	:param args: the command line arguments (as argparse.namespace object)
	:return: None
	"""

	nomlex_entries = json.load(args.lexicon[0])

	# Each input can be a string sentence or a name of file
	# Name of file is preferable
	inputs = []
	for input_str in args.input:
		if os.path.isfile(input_str):
			input_data = load_txt_file(input_str)
		else:
			input_data = [input_str]

		inputs.append(input_data)

	DictsAndTables.output_loc = args.output[0]

	num_of_checked_sentences = 0
	total_num_of_sentences = len(inputs[0])

	# Use a random order of the data (if needed)
	random_indexes = np.arange(len(inputs[0]))
	if DictsAndTables.shuffle_data: np.random.shuffle(random_indexes)

	# Verb module
	if args.verb:
		for i in random_indexes:
			arranged_print(inputs[0][i])
			seperate_line_print(verbal_to_nominal(nomlex_entries, inputs[0][i]))
			arranged_print("")

			num_of_checked_sentences += 1
			if not DictsAndTables.should_print_to_screen and total_num_of_sentences != 1:
				if num_of_checked_sentences == 1:
					print("Scanning " + str(num_of_checked_sentences) + "/" + str(total_num_of_sentences) + " sentences!")
				else:
					print("\033[1AScanning " + str(num_of_checked_sentences) + "/" + str(total_num_of_sentences) + " sentences!")

	# Nominalization module
	if args.nom:
		for i in random_indexes:
			if type(inputs[0][i]) == tuple: # Input is already parsed
				sent, dep = inputs[0][i]
				arranged_print(sent)
				seperate_line_print(extract_args_from_nominal(nomlex_entries, dependency_tree=dep))
			else: # Input is the sentence string
				arranged_print(inputs[0][i])
				seperate_line_print(extract_args_from_nominal(nomlex_entries, sent=inputs[0][i]))
			arranged_print("")

			num_of_checked_sentences += 1
			if not DictsAndTables.should_print_to_screen and total_num_of_sentences!= 1:
				if num_of_checked_sentences == 1:
					print("Scanning " + str(num_of_checked_sentences) + "/" + str(total_num_of_sentences) + " sentences!")
				else:
					print("\033[1AScanning " + str(num_of_checked_sentences) + "/" + str(total_num_of_sentences) + " sentences!")

	# Matching module
	if args.match:
		if len(inputs) < 2:
			print('error: input should contain two inputs in case of --match module')
			return

		# Counting the founded subcats so far
		DictsAndTables.subcats_counts = Counter()

		subcats = list(set([i[0] for i in get_comlex_table()] + ["NOM-INTRANS", "NOM-INTRANS-RECIP"]))
		with open(DictsAndTables.output_loc.name, 'r') as read_output_file:
			output_file_lines = read_output_file.readlines()

		for subcat in subcats:
			DictsAndTables.subcats_counts[subcat] = 0

		for line in output_file_lines:
			for subcat in subcats:
				if "'" + subcat + "'" in line:
					DictsAndTables.subcats_counts[subcat] += 1

		seperate_line_print(match_patterns(nomlex_entries, inputs[0], inputs[1]))

	DictsAndTables.output_loc.close()

if __name__ == '__main__':
	import sys

	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
									 description='This program has tree main modules:\n'
												 '1. Extracting arguments from sentences with a main verb, and translating the sentence to nominal form\n'
												 '\tExample- "[A0 IBM] appointed [A1 Alice Smith]" --> arguments of "appoint"\n'
												 '\t\tNote: brackets and arguments names are optional\n'
												 '2. Extracting arguments from sentences with nominalizations\n'
												 '\tExample- "Alice Smith\'s appointment by IBM" --> arguments of "appointment"\n'
												 '3. Finding matches between verbal sentences and nominal sentences\n'
												 '\tExample- "[A0 IBM] appointed [A1 Alice Smith]", "Alice Smith\'s appointment by IBM" --> match\n'
												 '\t\tNote: The order in important (verbal_sentences, verbal_sentences)')

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

	main(args)