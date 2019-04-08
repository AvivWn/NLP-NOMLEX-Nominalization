import sys
import os
sys.path.append(os.path.abspath(os.path.join('.')))
print(sys.path)

import MatchingPatterns
import Main
import re
from collections import defaultdict

def main(arguments):
	json_file_name, test_file_name = arguments

	nomlex_entries = Main.load_json_data(json_file_name)
	lines = open(test_file_name, "r+").readlines()
	data = []

	for line in lines:
		if line != "\n":
			line = line.replace("\n", "").replace("\"", "")
			splitted = re.split(r'\t+', line.rstrip('\t'))
			data.append((splitted[0], splitted[1], splitted[2]))

	statuses_counts = defaultdict()
	for subcat, verbal_sentence, nominal_sentence in data:
		matches, match_status = MatchingPatterns.match_patterns(nomlex_entries, verbal_sentence, nominal_sentence)

		if match_status in statuses_counts.keys():
			statuses_counts[match_status] += 1
		else:
			statuses_counts[match_status] = 1

		print(matches)
		print(match_status, "(" + subcat + ", '" + verbal_sentence + "', '" + nominal_sentence + "')")

	Main.seperate_line_print(dict(statuses_counts))




if __name__ == '__main__':
	"""
	Command line arguments-
		json_file_name test_file_name
	"""
	import sys

	main(sys.argv[1:])