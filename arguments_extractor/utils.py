import time
from collections import defaultdict

import spacy
from spacy.util import compile_infix_regex
from spacy.tokenizer import Tokenizer
from spacy.tokens import Token
import inflect

from arguments_extractor.constants.lexicon_constants import *
from arguments_extractor import config

engine = inflect.engine()

# Load the ud-parser
Token.set_extension("subtree_text", getter=lambda token: " ".join([node.text for node in token.subtree]))
Token.set_extension("subtree_indices", getter=lambda token: [node.i for node in token.subtree])
ud_parser = spacy.load("en_ud_model_lg")

# Change the tokenizer, so it won't split hypens between words
def custom_tokenizer(parser):
	# inf = list(parser.Defaults.infixes)            # Default infixes
	# inf.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")    # Remove the generic op between numbers or between a number and a -
	# inf = tuple(inf)                               # Convert inf to tuple
	# infixes = inf + tuple([r"(?<=[0-9])[+*^](?=[0-9-])", r"(?<=[0-9])-(?=-)"])  # Add the removed rule after subtracting (?<=[0-9])-(?=[0-9]) pattern
	infixes = parser.Defaults.infixes
	infixes = [x for x in infixes if '-|–|—|--|---|——|~' not in x] # Remove - between letters rule
	infix_re = compile_infix_regex(infixes)

	return Tokenizer(parser.vocab, prefix_search=parser.tokenizer.prefix_search,
					 			suffix_search=parser.tokenizer.suffix_search,
					 			infix_finditer=infix_re.finditer,
					 			token_match=parser.tokenizer.token_match,
					 			rules=parser.Defaults.tokenizer_exceptions)

ud_parser.tokenizer = custom_tokenizer(ud_parser)



def difference_list(first, second):
	return list(set(first) - set(second))

def reverse_dict(dictionary):
	return {value:key for key, value in dictionary.items()}

def filter_list(l, is_more_general, keys_func, largest=True, greedy=False):
	# Finds the objects in the given list with the highest number of keys
	# The resulted list shouldn't contain two objects that mean same thing

	if greedy:
		l.sort(key=lambda x: len(keys_func(x)), reverse=True)

	general_list = []

	for x in l:
		found_more_general = False

		if largest and len(keys_func(x)) != len(keys_func(l[0])):
			continue

		compared_list = general_list if greedy else l

		for other_x in compared_list:
			# Whether the other item is more general than the current one
			if is_more_general(x, other_x):
				found_more_general = True
				break

		if not found_more_general:
			general_list.append(x)

	return general_list

def aggregate_to_dict(list_of_dicts):
	# Transforms a list of dictionaries, into dictionary of lists
	total_dict = defaultdict(list)

	for dic in list_of_dicts:
		for k in dic:
			if type(dic[k]) == list:
				total_dict[k] += dic[k]
			else:
				total_dict[k].append(dic[k])

	return total_dict

def flatten(l):
	return [item for sublist in l for item in sublist]

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

def get_dependency_tree(sentence, disable=None):
	"""
	Returns the dependency tree of a given sentence
	:param sentence: a string sentence, or an already parsed sentence (Doc object)
	:param disable: names of pipes that should be disabled in the parsing process
	:return: the dependency tree of the sentence (a sequence of Spacy tokens = Doc)
	"""

	if type(sentence) != str:
		return sentence

	sentence = sentence.strip(" \t\r\n")

	if disable is None:
		disable = []

	return ud_parser(sentence, disable=disable)

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