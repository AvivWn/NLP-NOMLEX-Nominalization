import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))

import MatchingPatterns
import Main
import re

def main(arguments):
	json_file_name, test_file_name = arguments

	nomlex_entries = Main.load_json_data(json_file_name)
	lines = open(test_file_name, "r").readlines()
	data = []

	for line in lines:
		if line != "\n":
			line = line.replace("\n", "").replace("\"", "")
			splitted = re.split(r'\t+', line.rstrip('\t'))
			data.append((splitted[0], splitted[1], splitted[2]))

	for subcat, verbal_sentence, nominal_sentence in data:
		matches = MatchingPatterns.match_patterns(nomlex_entries, verbal_sentence, nominal_sentence)

		print("(" + subcat + ", '" + verbal_sentence + "', '" + nominal_sentence + "')")
		print(matches)
		print("")



if __name__ == '__main__':
	"""
	Command line arguments-
		json_file_name test_file_name
	"""
	import sys

	main(sys.argv[1:])