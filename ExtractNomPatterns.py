import json
import itertools
from collections import Counter

def load_json_data(json_file_name):
	"""
	Loads the data from a json file
	:param json_file_name: The name of the file that needed to be saved
	:return: The the json data (basically a dictionary object)
	"""

	with open(json_file_name) as inputfile:
		data = json.load(inputfile)

	return data


def get_pairs(entry):
	"""
	Returns the possible object and subject pairs for the given entry
	:param entry: a dictionary info of a specific nominalization
	:return: a list of all possible pairs for a specific nominalization entry (list of tuples)
	"""

	default_subjects = list(entry.get("VERB-SUBJ", {}).keys())
	default_subjects.append("by")
	pairs = []

	for subj in default_subjects:
		if subj == "NOT-PP-BY":
			default_subjects.remove("by")

	for subcat, subcat_info in entry.get("VERB-SUBC", {}).items():
		objects = list(subcat_info.get("OBJECT", {}).keys())
		subjects = list(subcat_info.get("SUBJECT", {}).keys())

		if subjects == []:
			subjects = default_subjects

		pairs += list(itertools.product(subjects, objects))

	return pairs


def extract_nom_patterns(json_file_name):
	"""
	Extracts all the nominalization patterns from the json file with the given name
	:param json_file_name: the name of the file that should be extracted
	:return: a counted nominalization patterns that can be found in the file
	"""

	entries = load_json_data(json_file_name)
	patterns = []

	for _, entry in entries.items():
		pairs = get_pairs(entry)
		patterns += pairs

	print(Counter(patterns))

	return Counter(patterns)


if __name__ == '__main__':
	"""
	Command line arguments-
		json_file_name
	"""
	import sys

	extract_nom_patterns(sys.argv[1])