import os
import json
import pickle
from collections import defaultdict

from spacy.tokens import Token, Doc
from tqdm import tqdm

from arguments_extractor.rule_based.lexical_entry import Entry
from arguments_extractor.rule_based.utils import get_argument_candidates
from arguments_extractor.utils import get_lexicon_path, difference_list
from arguments_extractor.constants.ud_constants import *
from arguments_extractor.constants.lexicon_constants import *
from arguments_extractor import config

class Lexicon:
	entries = defaultdict(Entry)
	is_verb: bool

	MAX_N_CANDIDATES = 5

	def __init__(self, lisp_file_name, is_verb=False):
		json_file_path = get_lexicon_path(lisp_file_name, "json", is_verb=is_verb, is_nom=not is_verb)
		pkl_file_path = get_lexicon_path(lisp_file_name, "pkl", is_verb=is_verb, is_nom=not is_verb)

		# Should we create the Lexicon classes again (pkl formated lexicon)?
		if config.LOAD_LEXICON and os.path.exists(pkl_file_path):
			with open(pkl_file_path, "rb") as input_file:
				loaded_lexicon = pickle.load(input_file)
				self.use_loaded_lexicon(loaded_lexicon)
		else:
			self.entries = defaultdict(Entry)
			self.is_verb = is_verb

			# Loading the lexicon from the json file with the given name
			with open(json_file_path, "r") as input_file:
				lexicon_json = json.load(input_file)

			# Adding each entry of the lexicon to this object
			for entry_word in tqdm(lexicon_json.keys(), "Modeling the lexicon", leave=False):
				self.entries[entry_word] = Entry(lexicon_json[entry_word], is_verb)

			# Update the next entry for the linked entry
			for entry_word in self.entries.keys():
				self.entries[entry_word].set_next(self)

			with open(pkl_file_path, 'wb') as output_file:
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

	def find_nom_types(self, suitable_verb):
		# Finds all the possible nom types for the nominalization that appropriate for this verb (based on nomlex)

		if self.is_verb:
			return []

		nom_types = set()
		for nom_entry in self.entries.values():
			if nom_entry.orth == DEFAULT_ENTRY:
				continue

			if nom_entry.verb.split("#")[0] == suitable_verb:
				nom_types.update([nom_entry.get_nom_type()])

		return list(nom_types)

	def find_verbs(self, complement_type):
		# Finds the verbs that can get the given complement type

		if self.is_verb:
			return []

		suitable_verbs = set()
		for nom_entry in self.entries.values():
			if nom_entry.orth == DEFAULT_ENTRY:
				continue

			if nom_entry.get_nom_type() == complement_type:
				suitable_verbs.update([nom_entry.verb.split("#")[0]])

			for subcat in nom_entry.subcats.values():
				if complement_type in subcat.arguments.keys():
					suitable_verbs.update([nom_entry.verb.split("#")[0]])
					break

		return list(suitable_verbs)

	def find(self, word: Token, using_default=False, nom_dictionary=None, limited_verbs=None):
		"""
		Finds the given word in this lexicon
		:param word: word token (spacy Token)
		:param using_default: whether to use the default entry in the lexicon all of the time, otherwise only whenever it is needed
		:param nom_dictionary: a dictionary object of all the known nominalizations
		:param limited_verbs: a list of limited verbs, which limits the predicates that their arguments will be extracted (optional)
		:return: the suitable entry in the lexicon, or None if it doesn't exist
		"""

		# To extract from the verb lexicon, the word must be a verb
		if self.is_verb and word.pos_ != UPOS_VERB:
			return None, None

		# To extract from the nom lexicon, the word must be a noun
		if not self.is_verb and word.pos_ != UPOS_NOUN:
			return None, None

		word_entry = self.entries[DEFAULT_ENTRY]
		suitable_verb = None

		word_forms = [word.orth_, word.orth_.lower(), word.lemma_]
		for word_form in word_forms:
			if word_form in difference_list(self.entries.keys(), [DEFAULT_ENTRY]):
				if not using_default:
					word_entry = self.entries[word_form]

					if self.is_verb:
						suitable_verb = word_entry.orth
					else: # Nominalization entry
						suitable_verb = word_entry.verb.split("#")[0]

				break

		if suitable_verb is not None and (limited_verbs is None or suitable_verb in limited_verbs):
			return word_entry, suitable_verb

		if nom_dictionary is None:
			return None, None

		if self.is_verb:
			suitable_verb = word.lemma_
		else:
			# Find suitable verb for known nominalizatios that don't appear in nomlex
			for word_form in word_forms:
				suitable_verb = nom_dictionary.get_suitable_verb(word_form)
				if suitable_verb is not None:
					break

		if suitable_verb is not None and (limited_verbs is None or suitable_verb in limited_verbs):
			return word_entry, suitable_verb

		return None, None


	@staticmethod
	def _update_unused_candidates(dependency_tree: Doc, token_candidates: list, predicate_token: Token, used_tokens: list, extraction: dict, specify_none=False):
		if not specify_none:
			return

		extraction[COMP_NONE] = []

		# Add any candidate that isn't in the used tokens to the NONE complement
		for unused_token in difference_list(token_candidates, used_tokens):
			if unused_token == predicate_token: # if the token is the predicate (the root)
				start_span_index = unused_token.i
				end_span_index = unused_token.i
			else:
				if unused_token.dep_ not in [URELATION_NMOD, URELATION_NSUBJ, URELATION_IOBJ, URELATION_DOBJ, URELATION_NMOD_POSS, URELATION_COMPOUND, URELATION_NSUBJPASS]:
					continue

				subtree_tokens = list(unused_token.subtree)
				start_span_index = subtree_tokens[0].i
				end_span_index = subtree_tokens[-1].i

			argument_span = dependency_tree[start_span_index: end_span_index + 1]
			extraction[COMP_NONE].append(argument_span)

	def extract_arguments(self, dependency_tree: Doc, min_arguments=0, using_default=False, arguments_predictor=None, specify_none=False, trim_arguments=True, nom_dictionary=None, limited_verbs=None):
		"""
		Extracts the arguments for any relevant word of the given sentence that appear in this lexicon
		:param dependency_tree: the appropriate dependency tree for a sentence
		:param min_arguments: the minimum number of arguments for any founed extraction (0 is deafult)
		:param using_default: whether to use the default entry in the lexicon all of the time, otherwise only whenever it is needed
		:param arguments_predictor: the model-based extractor object to determine the argument type of a span (optional)
		:param specify_none: wether to specify in the resulted extractions about the unused arguments
		:param trim_arguments: wether to trim the argument spans in the resulted extractions
		:param nom_dictionary: a dictionary object of all the known nominalizations (optional)
		:param limited_verbs: a list of limited verbs, which limits the predicates that their arguments will be extracted (optional)
		:return: all the founded argument extractions for any relevant word ({Token: [{COMP: Span}]})
		"""

		extractions_per_word = defaultdict(list)

		for token in dependency_tree:
			# Try to find an appropriate entry for the current word token
			word_entry, suitable_verb = self.find(token, using_default, nom_dictionary, limited_verbs)
			if word_entry is None:
				continue

			# Get the candidates for the arguments of this word (based on relevant direct links in the ud dependency tree)
			argument_candidates = get_argument_candidates(token, is_verb=self.is_verb)

			if len(argument_candidates) > self.MAX_N_CANDIDATES:
				continue

			# if config.DEBUG:
				# print(f"Candidates for {token.orth_}:", [candidate_token._.subtree_text if candidate_token != token else candidate_token.orth_ for candidate_token in argument_candidates])

			# Get all the possible extractions of this word
			extractions = word_entry.match_arguments(argument_candidates, token, suitable_verb, arguments_predictor)

			for extraction in extractions:
				if len(extraction.get_complements()) >= min_arguments:
					extraction_dict = extraction.as_span_dict(trim_arguments)
					self._update_unused_candidates(dependency_tree, argument_candidates, token, extraction.get_tokens(), extraction_dict, specify_none)
					extractions_per_word[token].append(extraction_dict)

			if config.DEBUG and len(extractions_per_word.get(token, [])) > 1:
				print(extractions_per_word[token])
				raise Exception("Found word with more than one legal extraction.")

		return extractions_per_word