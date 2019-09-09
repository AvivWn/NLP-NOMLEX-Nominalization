import json
import os
import pickle
import random
from itertools import chain, combinations
from collections import defaultdict

from ExtractNomlexPatterns import extract_nom_patterns, aggregate_patterns
from NominalPatterns import pattern_to_UD, extract_args_from_nominal, get_dependency
from DictsAndTables import get_all_of_noms
from NomlexExtractor import load_txt_file
import DictsAndTables

# Constants
NOM_FOR_TRAIN_PART = 0.8

def patterns_in_groups(unique_patterns):
	first_dep_links_dict = {}  # {'prep': 1, 'compound': 2, 'poss': 3, 'advmod': 4, 'acl': 5, 'amod': 6, 'advcl': 7}
	patterns_groups_dict = defaultdict(list)
	links_with_specific_value = []

	count = 0

	for pattern in unique_patterns:
		pattern_UD_list = pattern_to_UD(pattern)

		for pattern_UD in pattern_UD_list:
			for subentry, value in pattern_UD.items():
				if str(value[0]) not in first_dep_links_dict.keys():
					count += 1
					first_dep_links_dict[str(value[0])] = count

				if "_" in str(value[0]):
					splitted = str(value[0]).split("_")

					if splitted[0] not in links_with_specific_value:
						links_with_specific_value.append(splitted[0])

	print(links_with_specific_value)
	print(first_dep_links_dict)

	for pattern in unique_patterns:
		pattern_UD_list = pattern_to_UD(pattern)

		for pattern_UD in pattern_UD_list:
			first_values = []
			for subentry, value in pattern_UD.items():
				first_values.append(str(value[0]))

			first_value_trans = []
			for first_value in list(first_values):
				first_value_trans.append(first_dep_links_dict[first_value])

			first_value_trans = sorted(first_value_trans)
			new_first_value_trans = []
			for first_value in first_value_trans:
				new_first_value_trans.append(str(first_value))

			patterns_groups_dict["-".join(new_first_value_trans)] += [(pattern, pattern_UD)]

	return patterns_groups_dict, first_dep_links_dict, links_with_specific_value

def powerset(iterable):
	"powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
	s = list(iterable)
	return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def get_limited_patterns(dep, nom_idx, patterns_groups_dict, first_dep_links_dict, links_with_specific_value):
	first_dep_links_values = []

	# Moving over the dependency tree and finding all the first dependency links that leads to the nom in the given index
	for word in dep:
		if word[5] == nom_idx + 1:
			if word[6] in links_with_specific_value:
				tmp = "_".join([word[6], word[2]])

				if tmp in first_dep_links_dict.keys():
					first_dep_links_values.append(tmp)
			else:
				if word[6] in first_dep_links_dict.keys():
					first_dep_links_values.append(word[6])

	# Translating those links to number
	first_value_nums = []
	for first_value in first_dep_links_values:
		first_value_nums.append(str(first_dep_links_dict[first_value]))

	# Sort the list (according to the int values) and translate the values to
	first_value_nums = sorted(first_value_nums, key=float)

	# Save the patterns of each sub-group as the limited patterns
	limited_patterns = []
	for subset in powerset(first_value_nums):
		limited_patterns += patterns_groups_dict["-".join(subset)]

	return limited_patterns

def find_argument(sentence, argument, value, first_index, tags):
	sent_index = first_index
	arg_index = 0

	splited_argval = value.split(" ")
	splited_sent = sentence.split(" ")

	while arg_index < len(splited_argval) and sent_index < len(splited_sent):
		if splited_sent[sent_index] == splited_argval[arg_index]:
			# Adding the argument to the tags list
			if tags[sent_index] != "NONE":
				tags[sent_index] += "_" + argument.upper()
			else:
				tags[sent_index] = argument.upper()

			arg_index += 1

		sent_index += 1

def arguments_to_tags(sentence, splited_sentence, nom_index, arguments):
	tags = ["NONE"] * len(splited_sentence)
	tags[nom_index] = "NOM"
	found_not_none = False

	for argument, value in arguments.items():
		index = value[0]
		value = value[-1]

		if index != -1:
			find_argument(sentence, argument, value, index, tags)
			found_not_none = True

	return tags, found_not_none

def write_to_right_file(nom, train_noms, train_file, dev_file, text):
	print(text)
	return

	if nom in train_noms:
		train_file.write(text + "\n")
		train_file.flush()
	else:
		dev_file.write(text + "\n")
		dev_file.flush()

def tags_to_text(tags):
	tags += ["NONE"] # To make sure that also the last tag is used
	text_tuples = []
	last_tag = ("NONE", -1)

	for i in range(len(tags)):
		if tags[i] == "NONE":
			if tags[i] != last_tag[0]:
				text_tuples.append(str(last_tag[0]) + "_" + str(last_tag[1]) + "_" + str(i - 1))
				last_tag = ("NONE", i)
		else:
			if last_tag[0] == "NONE":
				last_tag = (tags[i], i)
			elif last_tag[0] != tags[i]:
				text_tuples.append(str(last_tag[0]) + "_" + str(last_tag[1]) + "_" + str(i - 1))
				last_tag = (tags[i], i)

	return " ".join(text_tuples)

def create_example(nomlex_entries, sentence, dep, train_noms, train_file, dev_file, limited_patterns_func):
	splited_sentence = sentence.split(" ")

	# Extracting all the arguments from all the nominalizations in the sentence, using the right patterns (for each nom)
	right_nom_arguments = extract_args_from_nominal(nomlex_entries, sent=sentence, dependency_tree=dep,
													keep_arguments_locations=True)

	# We must have right arguments in order to continue
	if right_nom_arguments == {} or all([value == [] for nom, value in right_nom_arguments.items()]):
		return

	# Extracting all the arguments from all the nominalizations in the sentence, using all the possible patterns
	all_nom_arguments = extract_args_from_nominal(nomlex_entries, sent=sentence, dependency_tree=dep,
												  keep_arguments_locations=True, get_all_possibilities=True,
												  limited_patterns_func=limited_patterns_func)

	found_any_arguments = False

	# Moving over all the nominalization that was found in both extraction
	# Of course that what was found in "right" was found also in "all"
	for nom, arguments_list in all_nom_arguments.items():
		right_tags_list = []

		# Check for the right arguments that extracted for that nom
		if nom in right_nom_arguments.keys():
			for arguments in right_nom_arguments[nom]:
				right_tags, found_not_none = arguments_to_tags(sentence, splited_sentence, nom[2], arguments)

				if found_not_none:
					right_tags_list.append(right_tags)

		# Check for all the arguments that extracted for that nom (using all patterns)
		for arguments in arguments_list:
			curr_tags, found_not_none = arguments_to_tags(sentence, splited_sentence, nom[2], arguments)

			if found_not_none:
				# Each argument list start with a sentence
				if not found_any_arguments:
					write_to_right_file(nom[0], train_noms, train_file, dev_file, "# " + sentence)
					found_any_arguments = True

				# Write that argument list to the right file
				if curr_tags in right_tags_list:
					write_to_right_file(nom[0], train_noms, train_file, dev_file, "+ " + tags_to_text(curr_tags))
				else:
					write_to_right_file(nom[0], train_noms, train_file, dev_file, "- " + tags_to_text(curr_tags))

def create_data(nomlex_filename, input_filename):
	DictsAndTables.should_clean = False

	# Loading the nomlex lexicon
	with open(nomlex_filename, "r") as nomlex_file:
		nomlex_entries = json.load(nomlex_file)

	DictsAndTables.all_noms, DictsAndTables.all_noms_backwards = get_all_of_noms(nomlex_entries)

	# Aggregating all the patterns in the lexicon
	unique_patterns = aggregate_patterns(extract_nom_patterns(nomlex_entries))

	# Creating a limited patterns func- a function that will return a limited number of patterns according to the sentence, dependency tree and nominalization index
	# Each pattern will include both the comlex and the ud dependency links version
	patterns_groups_dict, first_dep_links_dict, links_with_specific_value = patterns_in_groups(unique_patterns)
	limited_patterns_func = lambda dep, nom_idx: get_limited_patterns(dep, nom_idx, patterns_groups_dict, first_dep_links_dict, links_with_specific_value)

	# Splitting the NOMLEX nominalizations into train and dev noms
	noms = list(DictsAndTables.all_noms.keys())
	random.shuffle(noms)
	train_noms = noms[0: int(len(noms) * NOM_FOR_TRAIN_PART)]

	train_file = open("train4", "w+")
	dev_file = open("valid4", "w+")

	# Example
	#sentence = "The appointment of Alice by Apple"
	#dep = get_dependency(sentence)
	#create_example(nomlex_entries, sentence, dep, train_noms, train_file, dev_file, limited_patterns_func)

	# Loading the data
	if not os.path.exists(input_filename + "_as_list"):
		input_data = load_txt_file(input_filename)

		with open(input_filename + "_as_list", "wb") as patterns_file:
			pickle.dump(input_data, patterns_file)
	else:
		# Used the last saved file
		with open(input_filename + "_as_list", "rb") as patterns_file:
			input_data = pickle.load(patterns_file)

	i = 0

	# Moving over the sentences
	for sentence, dep in input_data:
		create_example(nomlex_entries, sentence, dep, train_noms, train_file, dev_file, limited_patterns_func)
		print(i)
		i += 1

	train_file.close()
	dev_file.close()



if __name__ == '__main__':
	import sys
	create_data(sys.argv[1], sys.argv[2])