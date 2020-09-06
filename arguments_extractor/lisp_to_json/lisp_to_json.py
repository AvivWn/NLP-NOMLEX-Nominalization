import os
import json

from tqdm import tqdm

from arguments_extractor.lisp_to_json.simplify_lexicon import simplify_lexicon
from arguments_extractor.lisp_to_json.utils import get_current_specs
from arguments_extractor.utils import get_lexicon_path
from arguments_extractor.constants.lexicon_constants import *
from arguments_extractor import config

# For debug
phrases = []

def parse_lines(lisp_file_name):
	"""
	Parses the line of the file with the given name (a file with lisp format)
	:param lisp_file_name: the file that is being parsed
	:return: a list of the lines of the file (only the relevant information)
	"""

	input_file = open(lisp_file_name, "r+")
	file_lines = input_file.readlines()
	lines = []

	# Moving over each line in the input file
	for line in tqdm(file_lines, "Parsing Lisp Lexicon", leave=False):
		if line != "\n":
			# Spacing up all the opening\closing brackets
			line = line.replace("(", " ( ").replace(")", " ) ").replace(") \n", ")\n") #.replace("\"", "").split(' ')

			word = ""

			# Dealing with a phrase of several words (seperated with spaces)
			is_between_brackets = False
			new_line = ""
			for char in line:
				if char == "\"":
					is_between_brackets = not is_between_brackets
					if " " in word and all([i.isalpha() or i == " " for i in word]): phrases.append(word)
					word = ""
				else:
					if is_between_brackets:
						word += char

					if char == " ":
						if is_between_brackets:
							char = '_'

				new_line += char

			temp_splitted_line = new_line.split(' ') #.replace("\"", "").split(' ')

			splitted_line = []

			for i in range(len(temp_splitted_line)):
				if temp_splitted_line[i] != '':
					splitted_line.append(temp_splitted_line[i].replace('_', ' ').replace('\n', ''))

			lines.append(splitted_line)

	return lines



def remove_quotes_around(string):
	# Remove the qoutes symbols from the start of the string
	if string.startswith('\"'):
		string = string[1:]

	# Remove the qoutes symbols from the end of the string
	if string.endswith('\"'):
		string = string[:-1]

	return string

def get_list(lines, index, in_line_index):
	"""
	Returns the data list that should appear from the given indexes
	:param lines: the list of data lines
	:param index: the current line index
	:param in_line_index: the current index in line
	:return: the updated indexes and a list of data that was found
	"""

	curr_list = []

	# Waiting to the relevant closing brackets sign
	while not lines[index][in_line_index] == ")":
		# There is a sub entry (another tag)
		if lines[index][in_line_index].startswith(":"):
			index, in_line_index, entry = translate_entry(lines, index, in_line_index)
			in_line_index -= 1
			curr_list.append(entry)

		# There is a sub-list
		elif lines[index][in_line_index] == "(":
			index, in_line_index, node = get_list(lines, index, in_line_index + 1)
			curr_list.append(node)

		# There is a string
		else:
			curr_list.append(remove_quotes_around(lines[index][in_line_index]))

		in_line_index += 1

		if in_line_index == len(lines[index]):
			index += 1
			in_line_index = 0

	return index, in_line_index, curr_list

def get_tag_info(lines, index, in_line_index):
	"""
	Returns the tag's info, the info after the tag's name
	:param lines: the list of data lines
	:param index: the current line index
	:param in_line_index: the current index in line
	:return: the updated indexes and the info that was found from the given indexes (the tag's info)
			 This info can be a list (recursion) or a string (end of recursion)
	"""

	# Checking if the info is a list
	if lines[index][in_line_index] == "(":
		index, in_line_index, curr_list = get_list(lines, index, in_line_index + 1)

		new_curr_list = curr_list
		
		if type(curr_list) == list and len(curr_list) > 0:
			if type(curr_list[0]) == list:
				if len(curr_list) == 1 and type(curr_list[0]) != list:
					new_curr_list = curr_list[0]
				else:
					new_curr_list = {}

					for sublist in curr_list:
						if type(sublist) == str:
							new_curr_list.update({sublist: {}})
						elif len(sublist) == 1:
							new_curr_list.update({sublist[0]: {}})
						else:
							new_curr_list.update({sublist[0]: sublist[1]})

		in_line_index += 1

		if in_line_index == len(lines[index]):
			index += 1
			in_line_index = 0

		return index, in_line_index, new_curr_list

	# Otherwise, the info must be a string
	curr_str = []

	# Getting all the string info
	while not lines[index][in_line_index].startswith(":") and not lines[index][in_line_index] == ")":
		curr_str.append(lines[index][in_line_index])

		in_line_index += 1

		if in_line_index == len(lines[index]):
			index += 1
			in_line_index = 0

	# Info as string
	tag_info = remove_quotes_around(" ".join(curr_str))

	if tag_info == "":
		raise Exception("Founded an empty value in the original lexicon.")

	return index, in_line_index, tag_info

def translate_entry(lines, index, in_line_index):
	"""
	Translates a nominalization entry (can be also a inner entry of sub-categorization) from lisp format to json format
	:param lines: the list of data lines
	:param index: the current line index
	:param in_line_index: the current index in line
	:return: the updated indexes and the new entry
	"""

	entry = {}

	if in_line_index > len(lines[index]):
		index += 1
		in_line_index = 0

	# Waiting to the relevant closing brackets sign
	while lines[index][in_line_index] != ")":
		# Getting the tag info, for each tag (":tag_name") in the entry
		if lines[index][in_line_index].startswith(":"):
			curr_tag = lines[index][in_line_index].replace(":", "")
			index, in_line_index, tag_info = get_tag_info(lines, index, in_line_index + 1)
			entry.update({curr_tag: tag_info})

	return index, in_line_index, entry

def translate(lines):
	"""
	Translates the given data lines (written in lisp format) to json format list
	:param lines: the list of data lines
	:return: the json format list
	"""

	entries = []
	index = 0
	in_line_index = 0
	count_founded_entries = {}
	count = 0

	# Moving over all the given line
	pbar = tqdm(desc="Lisp To Json", total=len(lines), leave=False)
	while index < len(lines):
		entry_type = lines[index][in_line_index + 1]

		# Each time, translating a specific nominalization entry of NOMLEX
		new_index, in_line_index, entry = translate_entry(lines, index, in_line_index + 2)

		# Only entries that starts with "NOM"
		if entry_type == "NOM": # or entry_type.startswith("NOM"):
			if ENT_ORTH in entry.keys():
				# Getting the orth of the word without the numbering
				not_numbered_orth = ''.join([i for i in entry[ENT_ORTH] if not i.isdigit()])
				entry[ENT_ORTH] = not_numbered_orth

				# Dealing with entries that appear with the same "ORTH" more than once
				# (each time with same or differnt entry type, like NOM, NOM-LIKE and more)
				entry["numbered_orth"] = not_numbered_orth
				if not_numbered_orth in count_founded_entries.keys():
					count_founded_entries[not_numbered_orth] += 1
				else:
					count_founded_entries[not_numbered_orth] = 1

				if count_founded_entries[not_numbered_orth] != 1:
					entry["numbered_orth"] = not_numbered_orth + "#" + str(count_founded_entries[not_numbered_orth])

				# Two worded verbs include also a particle (like "sponge off")
				# In such cases the particle should be removed from the verb form
				if " " in entry[ENT_VERB]:
					entry[ENT_VERB] = entry[ENT_VERB].split(" ")[0]

				entries.append(entry)

		pbar.update(new_index - index + 1)
		index = new_index + 1
		count += 1
		in_line_index = 0

	pbar.close()
	print("Total entries in nomlex:", count)

	return entries



def lisp_to_json(lisp_file_name):
	"""
	Translates a lisp format file into a json format file
	:param lisp_file_name: the name of the lisp file of the lexicon (only the file name, without its all path)
	:return:
	"""

	lisp_file_path = get_lexicon_path(lisp_file_name, "lisp")
	json_file_path = get_lexicon_path(lisp_file_name, "json")
	verb_json_file_path = get_lexicon_path(lisp_file_name, "json", is_verb=True)
	nom_json_file_path = get_lexicon_path(lisp_file_name, "json", is_nom=True)

	# Load the initial loaded lexicon if possible
	if config.LOAD_LEXICON and os.path.exists(json_file_path):
		with open(json_file_path, "r") as loaded_file:
			lexicon_data = json.load(loaded_file)

		print("Total loaded json entries:", len(lexicon_data.keys()))

	# Otherwise, create the json lexicon
	else:
		# Parsing the input file and getting the lines in it
		lines = parse_lines(lisp_file_path)

		# Translating the lines into entries as json format
		entries = translate(lines)

		lexicon_data = {}

		for entry in entries:
			numbered_orth = entry["numbered_orth"]
			del entry["numbered_orth"]
			lexicon_data.update({numbered_orth: entry})

		print("Total translated entries to json:", len(lexicon_data.keys()))

		# Writing the lexicon into an output file as a json format
		with open(json_file_path, 'w') as outfile:
			json.dump(lexicon_data, outfile)


	# Rearranging the lexicon and spliting verbs and nominalizations into different lexicons
	try:
		verbs_lexicon, noms_lexicon = simplify_lexicon(lexicon_data)
	except:
		print(f"There is a bug for the specifications- {get_current_specs()}")
		raise

	# Writing the verbs lexicon into an output file as a json format
	with open(verb_json_file_path, 'w') as outfile:
		json.dump(verbs_lexicon, outfile)

	# Writing the nominalizations lexicon into an output file as a json format
	with open(nom_json_file_path, 'w') as outfile:
		json.dump(noms_lexicon, outfile)