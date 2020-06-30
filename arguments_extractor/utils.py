import time
from collections import defaultdict

import spacy
from spacy.tokens import Token
import inflect

from arguments_extractor.constants.lexicon_constants import *
from arguments_extractor import config

engine = inflect.engine()

# Load the ud-parser
Token.set_extension("subtree_text", getter=lambda token: " ".join([node.text for node in token.subtree]))
Token.set_extension("subtree_indices", getter=lambda token: [node.i for node in token.subtree])
ud_parser = spacy.load("en_ud_model_lg")

def difference_list(first, second):
	return list(set(first) - set(second))

def reverse_dict(dictionary):
	return {value:key for key, value in dictionary.items()}

def timeit(method):
	def timed(*args, **kw):
		ts = time.time()
		result = method(*args, **kw)
		te = time.time()
		if 'log_time' in kw:
			name = kw.get('log_name', method.__name__.upper())
			kw['log_time'][name] = int((te - ts) * 1000)
		else:
			print('%r  %2.2f ms' % \
				  (method.__name__, (te - ts) * 1000))
			return result

	return timed



def list_to_regex(list_of_options, delimiter, start_constraint="", end_constraint=""):
	for i in range(len(list_of_options)):
		list_of_options[i] = start_constraint + list_of_options[i] + end_constraint

	regex_pattern = delimiter.join(list_of_options)

	return regex_pattern

def get_dependency_tree(sentence):
	"""
	Returns the dependency tree of a given sentence
	:param sentence: a string sentence
	:return: the dependency tree of the sentence (a list of doc = sequence of Spacy tokens)
	"""

	return ud_parser(sentence)

def get_lexicon_path(file_name, type_of_file, is_verb=False, is_nom=False):
	file_name = file_name.replace(".txt", "")
	lexicon_directory = ""

	if type_of_file == "json":
		lexicon_directory = config.JSON_DIR
	elif type_of_file == "pkl":
		lexicon_directory = config.PKL_DIR
	elif type_of_file == "lisp":
		lexicon_directory = config.LISP_DIR
		type_of_file = "txt"

	if is_verb:
		return lexicon_directory + file_name + "-verb." + type_of_file
	elif is_nom:
		return lexicon_directory + file_name + "-nom." + type_of_file
	else:
		return lexicon_directory + file_name + "." + type_of_file

def get_linked_arg(is_verb):
	if is_verb:
		return LINKED_VERB

	return LINKED_NOM



def separate_line_print(input_to_print, indent_level=0):
	indentation_str = ""
	for _ in range(indent_level):
		indentation_str += "  "

	if type(input_to_print) == list:
		for x in input_to_print:
			if type(x) == defaultdict:
				x = dict(x)
			print(str(indentation_str) + str(x))

	elif type(input_to_print) == dict or type(input_to_print) == defaultdict:
		for tag, x in input_to_print.items():
			if x != []: # Print only if it is not an empty list (meaning only if it is worth printing)
				print(str(indentation_str) + str(tag) + ": ")
				separate_line_print(x, indent_level + 2)