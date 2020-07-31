from copy import deepcopy

from spacy.tokens import Token

from arguments_extractor.rule_based.ud_translator import *
from arguments_extractor.constants.ud_constants import URELATION_ANY

def get_right_value(table, subcat_type, default=None, is_verb=False):
	if subcat_type not in table.keys():
		return default

	if is_verb:
		return deepcopy(table[subcat_type][0])

	return deepcopy(table[subcat_type][1])

def get_argument_candidates(referenced_token: Token, limited_relations=None, is_verb=False):
	argument_candidates = []

	for sub_token in referenced_token.subtree:
		if limited_relations is not None and sub_token.dep_ not in limited_relations:
			continue

		if sub_token.dep_ in LINK_TO_POS.keys() and sub_token.head.i == referenced_token.i:
			argument_candidates.append(sub_token)

	if not is_verb:
		argument_candidates.append(referenced_token)

	return argument_candidates

def relation_to_position(word_token, referenced_token, is_verb):
	if referenced_token == word_token:
		return [POS_NOM]

	dep_link = word_token.dep_
	positions = get_right_value(LINK_TO_POS, dep_link, default=[], is_verb=is_verb)

	return positions

def get_word_in_relation(referenced_word, original_relation, start_index=0):
	founded_token = None

	head_relation_info = original_relation.split("_")
	relation = head_relation_info[0]
	specific = head_relation_info[1] if len(head_relation_info) == 2 else None

	for child_token in referenced_word.children:
		if child_token.i < start_index:
			continue

		if not child_token.dep_.startswith(relation) and relation != URELATION_ANY:
			continue

		if specific is None:
			founded_token = child_token
			break

		if specific.islower() and child_token.orth_ == specific:
			founded_token = child_token
			break

		if not specific.islower() and child_token.tag_ == specific:
			founded_token = child_token
			break

	return founded_token

def check_relations(referenced_word, relations):
	for head_relation in relations:
		if get_word_in_relation(referenced_word, head_relation) is None:
			return False

	return True