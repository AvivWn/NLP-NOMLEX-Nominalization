import json
import os
import pickle
import random
from itertools import chain, combinations
from collections import defaultdict

import DictsAndTables
from ExtractNomlexPatterns import extract_nom_patterns, aggregate_patterns
from NominalPatterns import pattern_to_UD, extract_args_from_nominal, get_dependency
from DictsAndTables import get_all_of_noms
from NomlexExtractor import load_txt_file

# Constants
PART_OF_NOM_FOR_TRAIN = 0.8
LEARNING_FILES_LOCATION = "learning/"
MAX_SENT_SIZE = 150



def patterns_in_groups(unique_patterns):
	"""
	Spliting the given unique patterns into groups
	:param unique_patterns: a list of patterns (list of dictionaries)
	:return: a tuple of:
			 	patterns_groups_dict: a dictionary of groups of patterns ({string, like 1-2-3: [pattern]})
			 	initial_dep_links_dict: a dictionary of the initial dependency links in the arguments of the given patterns ({string: id (int)}
			 	links_with_specific_value: a list of dependency links with specific values (like prep_of)
	"""

	initial_dep_links_dict = {}  # Something like {'prep': 1, 'compound': 2, 'poss': 3, 'advmod': 4, 'acl': 5, 'amod': 6, 'advcl': 7, ....}
	patterns_groups_dict = defaultdict(list)
	links_with_specific_value = [] # This list is needed in order to know when the value is important and when it isn't

	count = 0

	# First, creating initial_dep_links_dict and links_with_specific_value
	for pattern in unique_patterns:
		pattern_UD_list = pattern_to_UD(pattern)

		for pattern_UD in pattern_UD_list:
			for subentry, value in pattern_UD.items():
				if str(value[0]) not in initial_dep_links_dict.keys():
					count += 1
					initial_dep_links_dict[str(value[0])] = count

				# The dep link has a specific value
				if "_" in str(value[0]):
					splitted = str(value[0]).split("_")

					if splitted[0] not in links_with_specific_value:
						links_with_specific_value.append(splitted[0])

	# Then, Splitting the patterns into groups based on initial_dep_links_dict
	for pattern in unique_patterns:
		pattern_UD_list = pattern_to_UD(pattern)

		for pattern_UD in pattern_UD_list:
			inital_dep_links_values = []

			# Getting the initial dependency links for the current ud pattern
			for subentry, value in pattern_UD.items():
				inital_dep_links_values.append(str(value[0]))

			# Translating those links to numbers
			inital_dep_links_nums = []
			for inital_dep_link in list(inital_dep_links_values):
				inital_dep_links_nums.append(str(initial_dep_links_dict[inital_dep_link]))

			# Sorting the list (according to the float values of the strings which are actually numbers)
			inital_dep_links_nums = sorted(inital_dep_links_nums, key=float)

			# Remembering the pattern in the suitable group (based on the initial dependency links needed for the current ud pattern)
			patterns_groups_dict["-".join(inital_dep_links_nums)] += [(pattern, pattern_UD)]

	return patterns_groups_dict, initial_dep_links_dict, links_with_specific_value

def powerset(iterable):
	"""
	Calculates the powerset of the given iterable
	:param iterable: an iterable, like list
	:return: a list of the powerset members
	"""

	"powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
	s = list(iterable)
	return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def get_limited_patterns(dep, nom_idx, patterns_groups_dict, initial_dep_links_dict, links_with_specific_value):
	"""
	Returns a list of patterns that are suitable for extracting arguments of the nominalization in the given index in the given dependecy
	:param dep: the dependency tree of the sentence (list of tuples)
	:param nom_idx: the index of a nominalization in the sentence
	:param patterns_groups_dict: a dictionary of groups of patterns ({string, like 1-2-3: [pattern]})
	:param initial_dep_links_dict: a dictionary of the initial dependency links in the arguments of the given patterns ({string: id (int)}
	:param links_with_specific_value: a list of dependency links with specific values (like prep_of)
	:return: a small list of patterns ([(pattern, pattern_UD)]
	"""

	inital_dep_links_values = []

	# Moving over the dependency tree and finding all the initial dependency links that leads to the nom in the given index
	for word in dep:
		if word[5] == nom_idx + 1:
			if word[6] in links_with_specific_value:
				tmp = "_".join([word[6], word[2]])

				if tmp in initial_dep_links_dict.keys():
					inital_dep_links_values.append(tmp)
			else:
				if word[6] in initial_dep_links_dict.keys():
					inital_dep_links_values.append(word[6])

	# Translating those links to numbers
	inital_dep_links_nums = []
	for inital_dep_link in inital_dep_links_values:
		inital_dep_links_nums.append(str(initial_dep_links_dict[inital_dep_link]))

	# Sorting the list (according to the float values of the strings which are actually numbers)
	inital_dep_links_nums = sorted(inital_dep_links_nums, key=float)

	# Saving the patterns of each sub-group as the limited patterns
	limited_patterns = []
	for subset in powerset(inital_dep_links_nums):
		limited_patterns += patterns_groups_dict["-".join(subset)]

	return limited_patterns



def find_argument(sentence, argument, value, first_index, tags):
	"""
	Finds the given value of the argument in the sentence, and updates the tags list as a results
	:param sentence: a sentence (string)
	:param argument: the argument name (string)
	:param value: the argument value, which appear in the sentence (string)
	:param first_index: the first index of the argument value in the sentence (int)
	:param tags: the tag of each word in the sentence (so far), to which argument each word belongs (list)
	:return: None
	"""

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
	"""
	Translates the arguments in the sentence into a list of tags
	:param sentence: a sentence (string)
	:param splited_sentence: the splitted sentence into words (list)
	:param nom_index: an index of nominalization in the sentence (int)
	:param arguments: the arguments of that nom ({argument: value})
	:return: list of tags (for each word in the sentence), whether or not a useful argument was found
	"""

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

def tags_to_text(tags):
	"""
	Translates a list of tags into a sentence
	:param tags: a list of tags (list)
	:return: a sentence
	"""

	tags += ["NONE"] # To make sure that also the last tag is also used
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

def write_to_right_file(nom, train_noms, train_file, dev_file, text):
	"""
	Writes the given text to the right file
	:param nom: a nominalization (string)
	:param train_noms: the nominalizations that should appear in training (and not validation) examples (list)
	:param train_file: the file for training examples (file)
	:param dev_file: the file for validation examples (file)
	:param text: the text to write to a file (string)
	:return: None
	"""

	if nom in train_noms:
		train_file.write(text + "\n")
		train_file.flush()
	else:
		dev_file.write(text + "\n")
		dev_file.flush()



def create_example(nomlex_entries, sentence, dep, train_noms, train_file, dev_file, limited_patterns_func):
	"""
	Creates the suitable examples for the given sentence
	This function will extract the right and wrong arguments of all the nominalizations in the sentence
	:param nomlex_entries: a dictionary of nominalizations, according to the NOMLEX lexicon
	:param sentence: a sentence (string)
	:param dep: the dependency tree of the sentence (list of tuples)
	:param train_noms: the nominalizations that should appear in training (and not validation) examples (list)
	:param train_file: the file for training examples (file)
	:param dev_file: the file for validation examples (file)
	:param limited_patterns_func: a function that can reduce the suitable patterns for each nominalization,
								  based on the dependency tree and the nominalization location (lambda function)
	:return: None
	"""

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

	# Moving over all the nominalization that was found in both extraction
	# Of course that what was found in "right" was found also in "all"
	for nom, arguments_list in all_nom_arguments.items():
		found_any_arguments = False
		wrong_founded_arguments_list = []
		right_founded_arguments_list = []
		right_tags_list = []

		arguments_list.sort(key=lambda x: len(x.keys()), reverse=True)

		# Check for the right arguments that extracted for that nom
		if nom in right_nom_arguments.keys():
			for arguments in right_nom_arguments[nom]:
				right_tags, found_not_none = arguments_to_tags(sentence, splited_sentence, nom[2], arguments)

				if found_not_none:
					right_tags_list.append(right_tags)

		# The tagging is useless without a right list of tags (from NOMLEX)
		if right_tags_list != []:
			# Check for all the arguments that extracted for that nom (using all patterns)
			for arguments in arguments_list:
				curr_tags, found_not_none = arguments_to_tags(sentence, splited_sentence, nom[2], arguments)

				if found_not_none:
					# Each list of arguments start with a sentence
					if not found_any_arguments:
						write_to_right_file(nom[0], train_noms, train_file, dev_file, "# " + sentence)
						found_any_arguments = True

					is_subset = False

					# Write that list of arguments to the right file
					if curr_tags in right_tags_list:
						# Checking if those arguments aren't a subset of any of the founded arguments (a bigger arguments list)
						for founded_args_items in right_founded_arguments_list:
							if all(item in founded_args_items for item in arguments.items()):
								is_subset = True
								break

						if not is_subset:
							write_to_right_file(nom[0], train_noms, train_file, dev_file, "+ " + tags_to_text(curr_tags))
							right_founded_arguments_list.append(arguments.items())
					else:
						# Checking if those arguments aren't a subset of any of the founded arguments (a bigger arguments list)
						for founded_args_items in wrong_founded_arguments_list:
							if all(item in founded_args_items for item in arguments.items()):
								is_subset = True
								break

						if not is_subset:
							write_to_right_file(nom[0], train_noms, train_file, dev_file, "- " + tags_to_text(curr_tags))
							wrong_founded_arguments_list.append(arguments.items())

def create_data(nomlex_file_loc, input_file_loc):
	"""
	Creates the data examples (for training), according to the sentence in the given input file
	:param nomlex_file_loc: a location of a json file with the entries of NOMLEX lexicon
	:param input_file_loc: a location of a txt file with sentences (which can be already parsed)
	:return: None
	"""

	DictsAndTables.should_clean = False

	# Loading the nomlex lexicon
	with open(nomlex_file_loc, "r") as nomlex_file:
		nomlex_entries = json.load(nomlex_file)

	DictsAndTables.all_noms, DictsAndTables.all_noms_backwards = get_all_of_noms(nomlex_entries)

	# Aggregating all the patterns in the lexicon
	unique_patterns = aggregate_patterns(extract_nom_patterns(nomlex_entries))

	# Creating a limited patterns func- a function that will return a limited number of patterns according to the sentence, dependency tree and nominalization index
	# Each pattern will include both the comlex and the ud dependency links version
	patterns_groups_dict, initial_dep_links_dict, links_with_specific_value = patterns_in_groups(unique_patterns)
	limited_patterns_func = lambda dep, nom_idx, nom: get_limited_patterns(dep, nom_idx, patterns_groups_dict, initial_dep_links_dict, links_with_specific_value)

	if not os.path.isdir(LEARNING_FILES_LOCATION):
		os.mkdir(LEARNING_FILES_LOCATION)

	# Getting the train_noms
	if not os.path.exists(LEARNING_FILES_LOCATION + "config"):
		# Splitting the NOMLEX nominalizations into train and dev noms
		noms = list(DictsAndTables.all_noms.keys())
		random.shuffle(noms)

		# Choosing the noms which will create the training examples
		train_noms = noms[0: int(len(noms) * PART_OF_NOM_FOR_TRAIN)]

		# Saving the train_noms in a config file
		with open(LEARNING_FILES_LOCATION + "config", "w") as config_file:
			config_file.write("\t".join(train_noms))
	else:
		# Loading the train_noms from the config file
		with open(LEARNING_FILES_LOCATION + "config", "r") as config_file:
			train_noms = config_file.readlines()[0].split("\t")

	input_file_name = input_file_loc.split("/")[-1]

	train_file = open(LEARNING_FILES_LOCATION + "train2_" + input_file_name, "w+")
	dev_file = open(LEARNING_FILES_LOCATION + "valid2_" + input_file_name, "w+")

	# Example
	#sentence = "The appointment of Alice by Apple"
	#dep = get_dependency(sentence)
	#create_example(nomlex_entries, sentence, dep, train_noms, train_file, dev_file, limited_patterns_func)

	# Loading the data
	if not os.path.exists(input_file_loc + "_as_list"):
		input_data = load_txt_file(input_file_loc)

		with open(input_file_loc + "_as_list", "wb") as patterns_file:
			pickle.dump(input_data, patterns_file)
	else:
		# Used the last saved file
		with open(input_file_loc + "_as_list", "rb") as patterns_file:
			input_data = pickle.load(patterns_file)

	i = 0

	# Moving over the sentences
	for x in input_data:
		# Is the data already parsed?
		if type(x) == tuple and len(x) == 2:
			sentence, dep = x
		else:
			# Otherwise, we will parse each sentence separately
			sentence = x
			dep = get_dependency(x)

		if len(sentence.split(" ")) <= MAX_SENT_SIZE:
			# Creating all the suitable examples to the current sentence
			create_example(nomlex_entries, sentence, dep, train_noms, train_file, dev_file, limited_patterns_func)

		print(i)
		i += 1

	train_file.close()
	dev_file.close()



if __name__ == '__main__':
	import sys
	create_data(sys.argv[1], sys.argv[2])