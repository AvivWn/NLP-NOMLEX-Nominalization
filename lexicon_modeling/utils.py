from utils import *
from .ud_translator import *

def get_right_value(table, subcat_type, default=None, is_verb=False):
	if subcat_type not in table.keys():
		return default

	if is_verb:
		return deepcopy(table[subcat_type][0])

	return deepcopy(table[subcat_type][1])

def get_argument_candidates(dependency_tree, reference_word_index):
	argument_candidates = []

	for word in dependency_tree:
		if word[WORD_DEP_LINK] in LINK_TO_POS.keys() and word[WORD_HEAD_ID] == reference_word_index:
			argument_candidates.append(word[WORD_INDEX])

	return argument_candidates