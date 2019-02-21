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

	# Getting the default subject roles
	default_subjects = list(entry.get("VERB-SUBJ", {"NOM-IS-SUBJ": {}}).keys())
	default_subjects.append("PP-BY")

	# Checking if NOT-PP-BY appeared
	for subj in default_subjects:
		if subj == "NOT-PP-BY":
			default_subjects.remove("PP-BY")
			default_subjects.remove("NOT-PP-BY")

	patterns = list(itertools.product(default_subjects, ["NONE"]))

	required_list = list(entry.get("REQUIRED", {}).keys())


	for subcat, subcat_info in entry.get("VERB-SUBC", {}).items():
		objects_subentry = subcat_info.get("OBJECT", {})
		subjects_subentry = subcat_info.get("SUBJECT", {})

		if objects_subentry != "NONE" and subjects_subentry != "NONE":
			objects = list(objects_subentry.keys())
			subjects = list(subjects_subentry.keys())

			subjects.append("PP-BY")
			# Checking if NOT-PP-BY appeared
			for subj in subjects:
				if subj == "NOT-PP-BY":
					subjects.remove("PP-BY")
					subjects.remove("NOT-PP-BY")

			if subjects == []:
				subjects = default_subjects

			if "SUBJECT" not in required_list:
				patterns += list(itertools.product(["NONE"], objects_subentry.keys()))

			if "OBJECT" not in required_list:
				patterns += list(itertools.product(subjects_subentry.keys(), ["NONE"]))

			patterns += list(itertools.product(subjects, objects))
		elif objects_subentry != "NONE":
			patterns += list(itertools.product(["NONE"], objects_subentry.keys()))
		elif subjects_subentry != "NONE":
			subjects = list(subjects_subentry.keys())

			subjects.append("PP-BY")
			# Checking if NOT-PP-BY appeared
			for subj in subjects:
				if subj == "NOT-PP-BY":
					subjects.remove("PP-BY")
					subjects.remove("NOT-PP-BY")

			patterns += list(itertools.product(subjects, ["NONE"]))

	return patterns


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