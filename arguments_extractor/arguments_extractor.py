from .lexicon_modeling.lexicon import Lexicon
from .lisp_to_json.lisp_to_json import lisp_to_json
from .utils import *

class ArgumentsExtractor:
	verb_lexicon: Lexicon
	nom_lexicon: Lexicon

	def __init__(self, lexicon_file_name):
		curr_dir = os.path.dirname(__file__)
		verb_json_file_path = get_lexicon_path(lexicon_file_name, "json", working_directory=curr_dir, is_verb=True)
		nom_json_file_path = get_lexicon_path(lexicon_file_name, "json", working_directory=curr_dir, is_nom=True)

		# Should we create the JSON formated lexicon again?
		if not (LOAD_LEXICON and os.path.exists(verb_json_file_path) and os.path.exists(nom_json_file_path)):
			lisp_to_json(lexicon_file_name)

		self.verb_lexicon = Lexicon(lexicon_file_name, is_verb=True)
		self.nom_lexicon = Lexicon(lexicon_file_name, is_verb=False)

	@staticmethod
	def clean_extractions(dependency_tree, extractions, as_indexes):
		"""
		Cleans the resulted extraction, deletes duplicates between arguments and translates arguments into phrases
		:param dependency_tree: The ud dependency tree of the sentence
		:param extractions: A list of all the possible extractions based on a specific word
		:param as_indexes: Whether the resulted arguments of each extraction will be written as string or as a list of indexes
		:return: None
		"""

		for extraction in extractions:
			tmp_extraction = deepcopy(extraction)

			for complement_type in tmp_extraction.keys():
				complement_root_index = tmp_extraction[complement_type]
				relevant_indexes = []

				# Find all the subtrees for that root (including the root word), that aren't another complement by themselves
				for other_word in dependency_tree:
					if other_word[WORD_HEAD_ID] == complement_root_index:
						if other_word[WORD_INDEX] not in tmp_extraction.values():
							relevant_indexes += other_word[WORD_SUB_TREE_INDEXES]

					elif other_word[WORD_INDEX] == complement_root_index:
						relevant_indexes += [other_word[WORD_INDEX]]

				if as_indexes:
					extraction[complement_type] = relevant_indexes
				else:
					extraction[complement_type] = " ".join([dependency_tree[word_index][WORD_TEXT] for word_index in relevant_indexes])

	def extract_arguments(self, sentence: str, as_indexes=False, include_verbs=True, include_noms=True):
		dependency_tree = get_dependency_tree(sentence)
		extractions_per_word = {}

		if include_verbs:
			extractions_per_word.update(self.verb_lexicon.extract_arguments(dependency_tree))

		if include_noms:
			extractions_per_word.update(self.nom_lexicon.extract_arguments(dependency_tree))

		for _, extractions in extractions_per_word.items():
			self.clean_extractions(dependency_tree, extractions, as_indexes)

		return extractions_per_word