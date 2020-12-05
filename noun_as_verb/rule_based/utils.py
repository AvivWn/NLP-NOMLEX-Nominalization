from spacy.tokens import Token

from noun_as_verb.rule_based.ud_translator import *
from noun_as_verb.constants.ud_constants import URELATION_ANY
from noun_as_verb.rule_based import ArgumentCandidate

def get_argument_candidates(referenced_token: Token, limited_relations=None, include_nom=False):
	argument_candidates = []

	for sub_token in referenced_token.children:
		if limited_relations is not None and sub_token.dep_ not in limited_relations:
			continue

		if sub_token.dep_ in LINK_TO_POS.keys():
			argument_candidates.append(ArgumentCandidate(sub_token, referenced_token))

	if include_nom:
		argument_candidates.append(ArgumentCandidate(referenced_token, referenced_token))

	return argument_candidates

def is_noun(token: Token):
	return token.pos_ == UPOS_NOUN

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