from copy import deepcopy

from spacy.tokens import Token

from arguments_extractor.rule_based.ud_translator import *

def get_right_value(table, subcat_type, default=None, is_verb=False):
	if subcat_type not in table.keys():
		return default

	if is_verb:
		return deepcopy(table[subcat_type][0])

	return deepcopy(table[subcat_type][1])

def get_argument_candidates(referenced_token: Token):
	argument_candidates = []

	for sub_token in referenced_token.subtree:
		if sub_token.dep_ in LINK_TO_POS.keys() and sub_token.head.i == referenced_token.i:
			argument_candidates.append(sub_token)

	return argument_candidates

def relation_to_position(word_info, is_verb):
	dep_link = word_info.dep_
	positions = get_right_value(LINK_TO_POS, dep_link, default=[], is_verb=is_verb)

	return positions

def check_relations(dependency_tree, referenced_word, relations):
	for head_relation in relations:
		found_relation = False

		for other_word in dependency_tree:
			head_relation_info = head_relation.split("_")
			relation = head_relation_info[0]

			if not other_word.head.i == referenced_word.i or not other_word.dep_ == relation:
				continue

			if len(head_relation_info) == 2:
				specific_word = head_relation_info[1]

				if other_word.orth_ == specific_word:
					found_relation = True
			else:
				found_relation = True

			if found_relation:
				break

		if not found_relation:
			return False

	return True