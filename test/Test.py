import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))

import MatchingPatterns
import Main
import re

def main(arguments):
	json_file_name, test_file_name = arguments

	nomlex_entries = Main.load_json_data(json_file_name)
	with open(test_file_name, "r") as test_file:
		lines = test_file.readlines()

	data = []

	for line in lines:
		if line != "\n":
			line = line.replace("\n", "").replace("\"", "")
			splitted = re.split(r'\t+', line.rstrip('\t'))
			data.append((splitted[0], splitted[1], splitted[2]))

	for subcat, verbal_sentence, nominal_sentence in data:
		_ = MatchingPatterns.match_patterns(nomlex_entries, [verbal_sentence], [nominal_sentence])



if __name__ == '__main__':
	"""
	Command line arguments-
		json_file_name test_file_name
	"""
	import sys

	main(sys.argv[1:])