import json

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
	for line in file_lines:
		if line != "\n":
			# Spacing up all the opening\closing brackets
			temp_splitted_line = line.replace("(", " ( ").replace(")", " ) ").replace(") \n", ")\n").replace("\"", "").split(' ')
			splitted_line = []

			for i in range(len(temp_splitted_line)):
				if temp_splitted_line[i] != '':
					splitted_line.append(temp_splitted_line[i].replace('\n', ''))

			lines.append(splitted_line)

	return lines



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
			curr_list.append(lines[index][in_line_index])

		in_line_index += 1

		if in_line_index == len(lines[index]):
			index += 1
			in_line_index = 0

	if curr_list[0] == "NONE":
		curr_list = curr_list[0]

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

	return index, in_line_index, " ".join(curr_str)

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

	# Moving over all the given line
	while index < len(lines):
		# Each time, translating a specific nominalization entry of NOMLEX
		index, in_line_index, entry = translate_entry(lines, index, in_line_index + 2)
		entries.append(entry)
		index += 1
		in_line_index = 0

	return entries



def lisp_to_json(lisp_file_name, json_file_name):
	"""
	Translates a lisp format file into a json format file
	:param lisp_file_name:
	:param json_file_name:
	:return:
	"""

	# Parsing the input file and getting the lines in it
	lines = parse_lines(lisp_file_name)

	# Translating the lines into entries as json format
	entries = translate(lines)

	data = {}

	for entry in entries:
		data.update({entry['ORTH']: entry})

	# Writing the data into a output file as a json format
	with open(json_file_name, 'w') as outfile:
		json.dump(data, outfile)

if __name__ == '__main__':
	"""
	Command line arguments- 
		lisp_file_name json_file_name
	"""
	import sys

	lisp_to_json(sys.argv[1], sys.argv[2])