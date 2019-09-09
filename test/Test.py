import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))

import MatchingPatterns
import NomlexExtractor
import re
import DictsAndTables
from DictsAndTables import get_all_of_noms, build_catvar_dict

def main(arguments):
	json_file_name, test_file_name = arguments

	nomlex_entries = NomlexExtractor.load_json_data(json_file_name)
	with open(test_file_name, "r") as test_file:
		lines = test_file.readlines()

	data = []

	for line in lines:
		if line != "\n":
			line = line.replace("\n", "").replace("\"", "")
			splitted = re.split(r'\t+', line.rstrip('\t'))
			data.append((splitted[0], splitted[1], splitted[2]))

	DictsAndTables.all_noms, DictsAndTables.all_noms_backwards = get_all_of_noms(nomlex_entries)
	DictsAndTables.catvar_dict = build_catvar_dict("../catvar_Data/catvar21.signed")

	noms_not_in_catvar = []
	for nom, nom_entry in nomlex_entries.items():
		if 'VERB' in nom_entry.keys():
			verb = nom_entry['VERB']
			clean_nom = DictsAndTables.all_noms[nom]
			if clean_nom not in DictsAndTables.catvar_dict.get(verb, []):
				noms_not_in_catvar.append((verb, nom))

	noms_not_in_nomlex = []
	for verb, noms in DictsAndTables.catvar_dict.items():
		for nom in noms:
			if nom not in DictsAndTables.all_noms_backwards.keys():
				noms_not_in_nomlex.append((verb, nom))

	print("Not in catvar:", len(noms_not_in_catvar))
	print("Not in NOMLEX:", len(noms_not_in_nomlex))

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