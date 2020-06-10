from lexicon_modeling.lexicon import Lexicon
from utils import *

class ArgumentsExtractor:
	verb_lexicon: Lexicon
	nom_lexicon: Lexicon

	def __init__(self, lexicon_filename):
		self.verb_lexicon = Lexicon(JSON_DIR + lexicon_filename.replace(".txt", "-verb.json"), is_verb=True)
		self.nom_lexicon = Lexicon(JSON_DIR + lexicon_filename.replace(".txt", "-nom.json"), is_verb=False)

	@staticmethod
	def clean_extractions(dependency_tree, extractions):
		"""
		Cleans the resulted extraction, deletes duplicates between arguments and translates arguments into phrases
		:param dependency_tree: The ud dependency tree of the sentence
		:param extractions: A list of all the possible extractions based on a specific word
		:return: None
		"""

		for extraction in extractions:
			tmp_extraction = deepcopy(extraction)

			for complement_type in extraction.keys():
				complement_root_index = tmp_extraction[complement_type]
				relevant_subtrees = []

				# Find all the subtrees for that root (including the root word), that aren't another complement by themselves
				for other_word in dependency_tree:
					if other_word[WORD_HEAD_ID] == complement_root_index:
						if other_word[WORD_INDEX] not in tmp_extraction.values():
							relevant_subtrees.append(other_word[WORD_SUB_TREE])

					elif other_word[WORD_INDEX] == complement_root_index:
						relevant_subtrees.append(other_word[WORD_TEXT])

				extraction[complement_type] = " ".join(relevant_subtrees)

	def extract_arguments(self, sentence: str):
		dependency_tree = get_dependency_tree(sentence)

		extractions_per_word = self.verb_lexicon.extract_arguments(dependency_tree)
		extractions_per_word.update(self.nom_lexicon.extract_arguments(dependency_tree))

		for word, extractions in extractions_per_word.items():
			self.clean_extractions(dependency_tree, extractions)

		return extractions_per_word