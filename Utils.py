from LexiconConstans import *
from UDTranslator import *
from config import *

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

def get_dependency_tree(sent):
	"""
	Returns the dependency tree of a given sentence
	:param sent: a string sentence
	:return: the dependency tree of the sentence (a list of tuples)
	"""

	dep = []

	# Here, the dependency tree is created using Spacy Package
	sentence_info = nlp(sent)
	for word_info in sentence_info:
		head_id = str(word_info.head.i)  # we want ids to be 1 based
		if word_info == word_info.head:  # and the ROOT to be 0.
			assert (word_info.dep_ == "ROOT"), word_info.dep_
			head_id = "0"  # root

		str_sub_tree = " ".join([node.text for node in word_info.subtree])

		dep.append({WORD_INDEX: word_info.i,
					WORD_TEXT: str(word_info.text),
					WORD_LEMMA: str(word_info.lemma_),
					WORD_POS_TAG: str(word_info.tag_),
					WORD_COARSE_POS_TAG: str(word_info.pos_),
					WORD_HEAD_ID: int(head_id),
					WORD_DEP_LINK: str(word_info.dep_),
					WORD_ENT_IOB_TAG: str(word_info.ent_iob_),
					WORD_ENT_TYPE: str(word_info.ent_type_),
					WORD_SUB_TREE: str_sub_tree})
	return dep

def arranged_print(input_to_print):
	if DEBUG:
		print(input_to_print)

def separate_line_print(input_to_print, indent_level=0):
	if DEBUG:
		indentation_str = ""
		for _ in range(indent_level):
			indentation_str += "  "

		if type(input_to_print) == list:
			for x in input_to_print:
				if type(x) == defaultdict:
					x = dict(x)
				arranged_print(str(indentation_str) + str(x))

		elif type(input_to_print) == dict or type(input_to_print) == defaultdict:
			for tag, x in input_to_print.items():
				if x != []: # Print only if it is not an empty list (meaning only if it is worth printing)
					arranged_print(str(indentation_str) + str(tag) + ": ")
					separate_line_print(x, indent_level + 2)