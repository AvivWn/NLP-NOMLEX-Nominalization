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


def get_patterns(entry):
	"""
	Returns the possible object and subject pairs for the given entry
	:param entry: a dictionary info of a specific nominalization
	:return: a list of all possible pairs for a specific nominalization entry (list of tuples)
	"""

	# Getting the default subject roles
	default_subjects = list(entry.get("VERB-SUBJ", {"NOM-IS-SUBJ": {}}).keys())
	default_subjects.append("PP-BY")

	# Checking if NOT-PP-BY appeared
	if "NOT-PP-BY" in default_subjects:
		default_subjects.remove("PP-BY")
		default_subjects.remove("NOT-PP-BY")

	patterns = []
	#patterns = list(itertools.product(default_subjects, ["NONE"], ["NONE"]))

	required_list = list(entry.get("REQUIRED", {}).keys())

	for subcat, subcat_info in entry.get("VERB-SUBC", {}).items():
		objects_subentry = subcat_info.get("OBJECT", {"NOM-IS-OBJ": {}})
		subjects_subentry = subcat_info.get("SUBJECT", {})
		ind_objects_subentry = subcat_info.get("IND-OBJ", {"NONE": {}})
		ind_objects = list(ind_objects_subentry.keys())

		if objects_subentry != "NONE" and subjects_subentry != "NONE":
			objects = list(objects_subentry.keys())
			subjects = list(subjects_subentry.keys())

			subjects.append("PP-BY")
			if "NOT-PP-BY" in subjects:
				subjects.remove("PP-BY")
				subjects.remove("NOT-PP-BY")

			if subjects == []:
				subjects = default_subjects

			if "SUBJECT" not in required_list:
				patterns += list(itertools.product(["NONE"], objects_subentry.keys(), ind_objects))

			if "OBJECT" not in required_list:
				patterns += list(itertools.product(subjects, ["NONE"], ind_objects))

			patterns += list(itertools.product(subjects, objects, ind_objects))
		elif objects_subentry != "NONE":
			patterns += list(itertools.product(["NONE"], objects_subentry.keys(), ind_objects))
		elif subjects_subentry != "NONE":
			subjects = list(subjects_subentry.keys())

			subjects.append("PP-BY")
			if "NOT-PP-BY" in subjects:
				subjects.remove("PP-BY")
				subjects.remove("NOT-PP-BY")

			patterns += list(itertools.product(subjects, ["NONE"], ind_objects))

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
		patterns += get_patterns(entry)

	patterns_counter = Counter(patterns)
	print(patterns_counter)
	print(len(patterns_counter))

	return patterns_counter


if __name__ == '__main__':
	"""
	Command line arguments-
		json_file_name
	"""
	import sys

	extract_nom_patterns(sys.argv[1])