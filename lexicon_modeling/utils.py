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
		if word[WORD_DEP_RELATION] in LINK_TO_POS.keys() and word[WORD_HEAD_ID] == reference_word_index:
			argument_candidates.append(word[WORD_INDEX])

	return argument_candidates

def relation_to_position(word_info, is_verb):
	"""
	Translates the
	:param word_info:
	:return:
	"""

	dep_link = word_info[WORD_DEP_RELATION]
	positions = get_right_value(LINK_TO_POS, dep_link, default=[], is_verb=is_verb)

	return positions

def check_relations(dependency_tree, referenced_word, relations):
	for head_relation in relations:
		found_relation = False

		for other_word in dependency_tree:
			head_relation_info = head_relation.split("_")
			relation = head_relation_info[0]

			if not other_word[WORD_HEAD_ID] == referenced_word[WORD_INDEX] or not other_word[WORD_DEP_RELATION] == relation:
				continue

			if len(head_relation_info) == 2:
				specific_word = head_relation_info[1]

				if other_word[WORD_TEXT] == specific_word:
					found_relation = True
			else:
				found_relation = True

			if found_relation:
				break

		if not found_relation:
			return False

	return True