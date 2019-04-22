import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))

import MatchingPatterns
import Main
import re
import DictsAndTables
from DictsAndTables import get_all_of_noms

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

	DictsAndTables.all_noms, DictsAndTables.all_noms_backwards = get_all_of_noms(nomlex_entries)

	num_of_exact_matches = 0
	total = len(data)
	for subcat, verbal_sentence, nominal_sentence in data:
		print(subcat, '"' + verbal_sentence + '"', '"' + nominal_sentence + '"')
		found_match = MatchingPatterns.match_patterns(nomlex_entries, [verbal_sentence], [nominal_sentence], exact_match=True)
		print("Found match?", found_match, "\n")
		num_of_exact_matches += found_match

	print("Found " + str(num_of_exact_matches) + " from " + str(total) + " sentences!")



if __name__ == '__main__':
	"""
	Command line arguments-
		json_file_name test_file_name
	"""
	import sys

	main(sys.argv[1:])