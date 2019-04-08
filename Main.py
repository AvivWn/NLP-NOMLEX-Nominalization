import json
from collections import defaultdict
from DictsAndTables import seperate_line_print, should_print
from VerbalPatterns import verbal_to_nominal
from NominalPatterns import extract_patterns_from_nominal
from MatchingPatterns import match_patterns


############################################### Loading and Saving ###############################################

def load_json_data(json_file_name):
	"""
	Loads the data from a json file
	:param json_file_name: The name of the file that needed to be loaded
	:return: The the json data (basically a dictionary object)
	"""

	with open(json_file_name) as inputfile:
		data = json.load(inputfile)

	return data

def load_txt_file(txt_file_name):
	"""
	Loads the data from a txt file with the given name
	:param txt_file_name: The name of the file that needed to be loaded
	:return: The the file data (as list of lines)
	"""
	with open(txt_file_name, "r+") as inputfile:
		data = inputfile.readlines()

	for i in range(len(data)):
		data[i] = data[i].replace("\n", "").replace("\r\n", "")

	return data



###################################################### Main ######################################################

def main(arguments):
	"""
	The main function
	:param arguments: the command line arguments
	:return: None
	"""

	# Extraction of patterns- from verbal sentence to nominal sentence (single case)
	if arguments[0] == "-patterns" and len(arguments) == 3:
		json_file_name = arguments[1]
		verbal_sent = arguments[2]

		nomlex_entries = load_json_data(json_file_name)
		seperate_line_print(verbal_to_nominal(nomlex_entries, verbal_sent))

	# Extraction of arguments- from verbal sentence to nominal sentence (multiple case- from file)
	elif arguments[0] == "-fpatterns" and len(arguments) == 3:
		json_file_name = arguments[1]
		verbal_sents_file_name = arguments[2]

		nomlex_entries = load_json_data(json_file_name)
		verbal_sents = load_txt_file(verbal_sents_file_name)

		for verbal_sent in verbal_sents:
			seperate_line_print(verbal_to_nominal(nomlex_entries, verbal_sent))

	# Extraction of arguments- from nominal sentence to verbal sentence (single case)
	elif arguments[0] == "-args" and len(arguments) == 3:
		json_file_name = arguments[1]
		nominal_sent = arguments[2]

		nomlex_entries = load_json_data(json_file_name)
		seperate_line_print(extract_patterns_from_nominal(nomlex_entries, nominal_sent))

	# Extraction of arguments- from nominal sentence to verbal sentence (multiple case- from file)
	elif arguments[0] == "-fargs" and len(arguments) == 3:
		json_file_name = arguments[1]
		nominal_sents_file_name = arguments[2]

		nomlex_entries = load_json_data(json_file_name)
		nominal_sents = load_txt_file(nominal_sents_file_name)

		for nominal_sent in nominal_sents:
			seperate_line_print(extract_patterns_from_nominal(nomlex_entries, nominal_sent))

	# Matching arguments extracted in verbal and nominal sentences (single case)
	elif arguments[0] == "-match" and len(arguments) == 4:
		json_file_name = arguments[1]
		verbal_sent = arguments[2]
		nominal_sent = arguments[3]

		nomlex_entries = load_json_data(json_file_name)
		seperate_line_print(match_patterns(nomlex_entries, verbal_sent, nominal_sent))

	# Matching arguments extracted in verbal and nominal sentences (multiple case- from file)
	elif arguments[0] == "-fmatch" and len(arguments) == 4:
		json_file_name = arguments[1]
		verbal_sents_file_name = arguments[2]
		nominal_sents_file_name = arguments[3]

		nomlex_entries = load_json_data(json_file_name)
		verbal_sents = load_txt_file(verbal_sents_file_name)
		nominal_sents = load_txt_file(nominal_sents_file_name)

		statuses_counts = defaultdict()
		for verbal_sent in verbal_sents:
			for nominal_sent in nominal_sents:
				matches, status = match_patterns(nomlex_entries, verbal_sent, nominal_sent)

				if should_print: print(status, "(" + verbal_sent + "', '" + nominal_sent + "')")
				seperate_line_print(matches)

				if status not in statuses_counts.keys():
					statuses_counts[status] = (0, 0)

				# Calculating also the new average
				statuses_counts[status] = (statuses_counts[status][0] + 1, (statuses_counts[status][0] + 1) / len(nominal_sents))

		seperate_line_print(dict(statuses_counts))

if __name__ == '__main__':
	"""
	Command line arguments-
		 -patterns json_file_name verbal_sentence (single)
		 -fpatterns json_file_name verbal_sentences_file_name (multiple)
		 -args json_file_name nominal_sentence (single)
		 -fargs json_file_name nominal_sentences_file_name (multiple)
		 -match json_file_name verbal_sentence nominal_sentence (single)
		 -fmatch json_file_name verbal_sentences_file_name nominal_sentences_file_name (multiple)
	"""
	import sys

	main(sys.argv[1:])