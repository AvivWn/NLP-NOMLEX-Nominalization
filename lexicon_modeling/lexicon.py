from .entry import *

class Lexicon:
	entries = defaultdict(Entry)
	is_verb: bool

	def __init__(self, json_filename, is_verb=False):
		lexicon_name = json_filename.split("/")[-1]
		print(json_filename)

		if LOAD_LEXICON and os.path.exists(PKL_DIR + lexicon_name + '.pkl'):
			with open(PKL_DIR + lexicon_name.replace(".json",'.pkl'), 'rb') as input_file:
				loaded_lexicon = pickle.load(input_file)
				self.use_loaded_lexicon(loaded_lexicon)
		else:
			self.entries = defaultdict(Entry)
			self.is_verb = is_verb

			# Loading the lexicon from the json file with the given name
			with open(json_filename, "r") as input_file:
				lexicon_json = json.load(input_file)

			# Adding each entry of the lexicon to this object
			for entry_word in lexicon_json.keys():
				self.entries[entry_word] = Entry(lexicon_json[entry_word], is_verb)

			# Update the next entry for the linked entry
			for entry_word in self.entries.keys():
				self.entries[entry_word].set_next(self)

			with open(PKL_DIR + lexicon_name.replace(".json",'.pkl'), 'wb') as output_file:
				pickle.dump(self, output_file)

	def get_entry(self, entry_word):
		if entry_word == "" or entry_word is None:
			return None

		if entry_word not in self.entries.keys():
			raise Exception(f"The word {entry_word} do not appear in this lexicon!")

		return self.entries[entry_word]

	def get_entries(self):
		return self.entries

	def is_verbal_lexicon(self):
		return self.is_verb

	def use_loaded_lexicon(self, loaded_lexicon):
		self.entries = loaded_lexicon.get_entries()
		self.is_verb = loaded_lexicon.is_verbal_lexicon()

	def find(self, word):
		"""
		Finds the given word in this lexicon
		:param word: a string word
		:return: the suitable word in the lexicon, or None otherwise
		"""

		if self.is_verb and word[WORD_UPOS_TAG] != UPOS_VERB:
			return None

		if not self.is_verb and word[WORD_UPOS_TAG] != UPOS_NOUN:
			return None

		if word[WORD_TEXT] in self.entries.keys():
			return word[WORD_TEXT]

		if word[WORD_TEXT].lower() in self.entries.keys():
			return word[WORD_TEXT].lower()

		if word[WORD_LEMMA] in self.entries.keys():
			return word[WORD_LEMMA]

		return None

	def extract_arguments(self, dependency_tree: list):
		"""
		Extracts the arguments for any word of the given sentence that appear in this lexicon
		:param dependency_tree: the appropriate dependency tree for a sentence
		:return: all the founded argument extractions for any word ({word: [ARG: root_index]})
		"""

		extractions_per_word = defaultdict(list)

		for word in dependency_tree:
			lexical_word = self.find(word)

			# Ignore words that don't appear in the lexicon
			if lexical_word is None:
				continue

			# Get the candidates for the arguments of this word (relevant direct links in the ud)
			argument_candidates = get_argument_candidates(dependency_tree, word[WORD_INDEX])
			print(f"Candidates for {word[WORD_TEXT]}:", [(dependency_tree[candidate_idx][WORD_SUB_TREE]) for candidate_idx in argument_candidates])

			# Get all the possible extractions of this word
			extractions = self.entries[lexical_word].match_arguments(dependency_tree, argument_candidates, word[WORD_INDEX])
			extractions_per_word[(word[WORD_TEXT], word[WORD_INDEX])] += extractions

		return extractions_per_word