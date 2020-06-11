from .subcat import *

class Entry:
	orth: str
	subcats = defaultdict(Subcat)
	next = None
	plural = None
	nom = None
	singular = None
	plural_freq = None
	is_verb: bool



	def __init__(self, entry: dict, is_verb):
		self.orth = entry[ENT_ORTH]
		self.nom_type = entry.get(ENT_NOM_TYPE, {})

		self.subcats = defaultdict(Subcat)
		for subcat_type in entry.get(ENT_VERB_SUBC, {}).keys():
			self.subcats[subcat_type] = Subcat(entry[ENT_VERB_SUBC][subcat_type], is_verb)

		self.next = entry.get(ENT_NEXT, None)
		self.plural = entry.get(ENT_PLURAL, None)
		self.nom = entry.get(ENT_NOM, None)
		self.singular = entry.get(ENT_SINGULAR, None)
		self.plural_freq = entry.get(ENT_PLURAL_FREQ, None)
		self.is_verb = is_verb

	def set_next(self, lexicon):
		self.next = lexicon.get_entry(self.next)

	def match_arguments(self, dependency_tree: list, argument_candidates: list, referenced_word_index: int):
		"""
		Matches the given argument candidates to the possible arguments of all the entries with the same orth (this, next and so on)
		:param dependency_tree: the appropriate dependency tree for a sentence
		:param argument_candidates: the candidates for the arguments of this entry (as list of root indexes)
		:return: a list of all the founded argument matching for this entry ([{ARG: root_index}])
		"""

		matchings = []

		# Match the arguments based on each subcat for this word entry
		for subcat_type in self.subcats.keys():
			matchings += self.subcats[subcat_type].match_arguments(dependency_tree, argument_candidates, referenced_word_index)

		# Match arguments also based on the "next" entry in the lexicon
		# Meaning, the aruguments properties of the same word with another sense
		if self.next is not None:
			matchings += self.next.match_arguments(dependency_tree, argument_candidates, referenced_word_index)

		# Sort the matchings based on the number of arguments
		matchings = sorted(matchings, key=lambda k: len(k.keys()), reverse=True)

		# Find only the unique matchings, which aren't sub-matchings
		unique_matchings = []
		for matching in deepcopy(matchings):
			is_sub_matching = any([matching.items() <= other_matching.items() for other_matching in unique_matchings])

			if not is_sub_matching:
				unique_matchings.append(matching)

		return unique_matchings