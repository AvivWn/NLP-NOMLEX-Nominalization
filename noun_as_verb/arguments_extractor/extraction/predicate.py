from spacy.tokens import Token

from noun_as_verb.lexicon_representation.utils import is_noun, is_verb


class Predicate:
	token: Token
	suitable_verb: str

	def __init__(self, token):
		self.token = token

	def get_token(self):
		return self.token

	def is_noun(self):
		return is_noun(self.token)

	def is_verb(self):
		return is_verb(self.token)
