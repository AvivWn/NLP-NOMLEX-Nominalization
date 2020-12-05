from copy import deepcopy

from spacy.tokens import Token

from .ud_translator import LINK_TO_POS
from noun_as_verb.constants.lexicon_constants import *
from noun_as_verb.constants.ud_constants import UPOS_VERB


class Candidate:
	argument_token: Token
	positions: []
	unrelevant_positions: []

	def __init__(self, argument_token: Token, referenced_token: Token):
		self.argument_token = argument_token
		self.positions = self.relation_to_position(referenced_token)
		self.irelevant_positions = []

	def relation_to_position(self, referenced_token):
		if referenced_token == self.argument_token:
			return [POS_NOM]

		is_verb = referenced_token.pos_ == UPOS_VERB

		dep_link = self.argument_token.dep_
		verb_pos, noun_pos = LINK_TO_POS.get(dep_link, [])
		positions = verb_pos if is_verb else noun_pos

		return deepcopy(positions)

	def get_token(self):
		return self.argument_token

	def get_possible_positions(self):
		return self.positions

	# def add_irrelevant_positions(self, position, prefixes):
	# 	self.irelevant_positions[prefixes)