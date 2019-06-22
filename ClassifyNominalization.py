import json
import os
import pickle
import numpy as np
from ExtractNomlexPatterns import extract_nom_patterns
from collections import Counter
from collections import defaultdict
from NominalPatterns import get_dependency, extract_argument, pattern_to_UD
from DictsAndTables import get_all_of_noms
from Main import load_txt_file

def suitable_patterns(patterns, word_index, dependency):
	found_argument = defaultdict(bool)
	suitable_patterns = []

	for pattern in patterns:
		ud_patterns_list = pattern_to_UD(pattern)

		for ud_pattern in ud_patterns_list:
			count_of_founded_args = 0

			if ud_pattern != {}:
				for arg, dep_links in ud_pattern.items():
					known_result = found_argument.get(str(dep_links), "None")

					if known_result == "None":
						arguments_list = extract_argument(dependency, dep_links, word_index)

						if arguments_list != []:
							found_argument[str(dep_links)] = True
							count_of_founded_args += 1
						else:
							found_argument[str(dep_links)] = False
							break

					elif known_result == True:
						count_of_founded_args += 1
					else:
						break

				if count_of_founded_args == len(dict(ud_pattern).keys()):
					suitable_patterns.append(ud_pattern)

	return suitable_patterns

def is_nom(patterns, sentence, word_index, dependency=None):
	if dependency is None:
		print(2)
		dependency = get_dependency(sentence)

	found_argument = defaultdict(bool)

	"""
	# Check nominalization only to nouns
	if dependency[word_index][4] == "NOUN" and dependency[word_index][8] == "":
		if len(suitable_patterns(patterns, word_index, dependency)) != 0:
			return True
	"""

	count = 0

	if dependency[word_index][4] == "NOUN" and dependency[word_index][8] == "":
		for pattern in patterns:
			ud_patterns_list = pattern_to_UD(pattern)

			for ud_pattern in ud_patterns_list:
				count_of_founded_args = 0

				if ud_pattern != {}:
					count += 1

					for arg, dep_links in ud_pattern.items():
						known_result = found_argument.get(str(dep_links), "None")

						if known_result == "None":
							arguments_list = extract_argument(dependency, dep_links, word_index)

							if arguments_list != []:
								found_argument[str(dep_links)] = True
								count_of_founded_args += 1
							else:
								found_argument[str(dep_links)] = False
								break

						elif known_result == True:
							count_of_founded_args += 1
						else:
							break

					if count_of_founded_args == len(dict(ud_pattern).keys()):
						print(ud_pattern)
						return True

	print(count)

	return False


def aggregate_patterns(nomlex_patterns_dict):
	unique_patterns = []
	ids_counter = Counter()
	pattern2id = {}

	for _, possible_arguments in nomlex_patterns_dict.values():
		for args_dict in possible_arguments:
			del args_dict["verb"]
			del args_dict["subcat"]

			if args_dict != {}:
				if args_dict not in unique_patterns:
					unique_patterns.append(args_dict)
					pattern2id[len(ids_counter)] = args_dict
					ids_counter[len(ids_counter)] = 1
				else:
					ids_counter[unique_patterns.index(args_dict)] += 1

	for key, count in ids_counter.most_common(10):
		print(str(dict(pattern2id[key])) + " : " + str(count))

	print("... : ...")
	least_common_key, least_common_count = ids_counter.most_common()[-1]
	print(str(dict(pattern2id[least_common_key])) + " : " + str(least_common_count))
	print("total : " + str(len(unique_patterns)))

	return unique_patterns

def main(nomlex_filename, input_filename):
	if not os.path.exists(input_filename + "_as_list"):
		input_data = load_txt_file(input_filename)

		with open(input_filename + "_as_list", "wb") as patterns_file:
			pickle.dump(input_data, patterns_file)
	else:
		# Used the last saved file
		with open(input_filename + "_as_list", "rb") as patterns_file:
			input_data = pickle.load(patterns_file)

	with open(nomlex_filename, "r") as nomlex_file:
		nomlex_entries = json.load(nomlex_file)
	unique_patterns = aggregate_patterns(extract_nom_patterns(nomlex_entries))
	all_noms = get_all_of_noms(nomlex_entries)[0].values()

	#if not os.path.exists(nomlex_filename + "_unique_patterns"):
	#	with open(nomlex_filename, "r") as nomlex_file:
	#		nomlex_entries = json.load(nomlex_file)
	#	unique_patterns = aggregate_patterns(extract_nom_patterns(nomlex_entries))
	#with open(nomlex_filename + "_unique_patterns", "wb") as nomlex_patterns_file:
	#	pickle.dump(unique_patterns, nomlex_patterns_file)
	#else:
	#	with open(nomlex_filename + "_unique_patterns", "rb") as nomlex_patterns_file:
	#		unique_patterns = pickle.load(nomlex_patterns_file)

	#print(is_nom(unique_patterns, "IBM's appointment of Alice.", 2))

	random_indexes = np.arange(len(input_data))
	np.random.shuffle(random_indexes)
	count_sentences = 0

	for input_index in random_indexes:
		sentence, dep = input_data[input_index]

		words = sentence.split(" ")
		for word_index in range(len(words)):
			# Check if the word is nominalization
			if is_nom(unique_patterns, sentence, word_index, dependency=dep): #and words[word_index] not in all_noms:
				if dep[word_index][2] not in all_noms:
					print(count_sentences, words[word_index], sentence)
				else:
					print(count_sentences, "*" + words[word_index] + "*", sentence)

		count_sentences += 1

if __name__ == '__main__':
	import sys

	sentence = "after several months of appeals for help the danish legation in russia issued olga a passport which she used to enter germany on the eve of its defeat eventually joining her eldest_son and his family in switzerland in early 1919"
	dep = get_dependency(sentence)
	dep_links = ['prep_after', ['pobj']]
	word_index = 9
	print(extract_argument(dep, dep_links, word_index))

	main(sys.argv[1], sys.argv[2])