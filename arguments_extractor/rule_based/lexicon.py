import os
import json
import pickle
from collections import defaultdict

from spacy.tokens import Token, Doc
from tqdm import tqdm

from arguments_extractor.rule_based.lexical_entry import Entry
from arguments_extractor.rule_based.utils import get_argument_candidates
from arguments_extractor.rule_based.extracted_argument import ExtractedArgument
from arguments_extractor.utils import get_lexicon_path, difference_list, get_linked_arg
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

		return self.entries.get(entry_word, None)

	def get_entries(self):
		return self.entries

	def is_verbal_lexicon(self):
		return self.is_verb

	def use_loaded_lexicon(self, loaded_lexicon):
		self.entries = loaded_lexicon.get_entries()
		self.is_verb = loaded_lexicon.is_verbal_lexicon()

	def get_types_per_verb(self, ignore_part=False):
		# Finds all the possible nom-types that are appropriate for each verb

		if self.is_verb:
			return {}

		nom_types_per_verb = defaultdict(set)
		for nom_entry in self.entries.values():
			if nom_entry.orth == DEFAULT_ENTRY:
				continue

			suitable_verb = nom_entry.get_suitable_verb()
			nom_types = nom_entry.get_nom_types(ignore_part=ignore_part)
			nom_types_per_verb[suitable_verb].update(nom_types)

		return nom_types_per_verb

	def get_verbs_by_arg(self, complement_type):
		# Finds the verbs that can be complemented by the given argument type

		suitable_verbs = set()
		for entry in self.entries.values():
			if entry.orth == DEFAULT_ENTRY:
				continue

			suitable_verb = entry.get_suitable_verb()

			# The complement type is the nom
			if complement_type in entry.get_nom_types(ignore_part=True):
				suitable_verbs.update([suitable_verb])

			# The argument of that type can complement the verb\nom
			for subcat in entry.subcats.values():
				if complement_type in subcat.arguments.keys():
					suitable_verbs.update([suitable_verb])
					break

		return list(suitable_verbs)



	def find(self, word:Token, using_default=False, verb_noun_matcher=None, limited_verbs=None, be_certain=False):
		"""
		Finds the given word in this lexicon
		:param word: word token (spacy Token)
		:param using_default: whether to use the default entry in the lexicon all of the time, otherwise only whenever it is needed
		:param verb_noun_matcher: a object that matches a verb with nouns whitin its word family using CATVAR (optional)
		:param limited_verbs: a list of limited verbs, which limits the predicates that their arguments will be extracted (optional)
		:param be_certain: whether to find only nominalizations that cannot be common nouns (not relevant for verbs)
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

		# The original form is used for nouns (cause we want plural to still be checked as plural)
		# For verbs, we use the lemma form (cause it might be finite)
		word_form = word.lemma_.lower() if self.is_verb else word.orth_.lower()

		# Try to use NOMLEX first
		if word_form in self.entries.keys() and not using_default:
			word_entry = self.entries[word_form]
			suitable_verb = word_entry.get_suitable_verb()

		found_verb = lambda verb: verb and (not limited_verbs or verb in limited_verbs)

		# Otherwise use the verb-noun matcher
		if using_default or (verb_noun_matcher and not found_verb(suitable_verb)):
			suitable_verb = verb_noun_matcher.get_suitable_verb(word)

		# Did we find an appropriate predicate?
		if found_verb(suitable_verb) and word_entry.must_be_predicate(word, be_certain):
			return word_entry, suitable_verb

		return None, None

	def _predicate_wrapper(self, word:Token, word_entry, arguments_predictor=None):
		noun_type = None
		entropy = None

		if not self.is_verb and arguments_predictor:
			noun_type. entropy = arguments_predictor.determine_noun_type(word)
			noun_type = noun_type if noun_type != COMP_NONE else None

		predicate = ExtractedArgument(word, word_entry, get_linked_arg(self.is_verb), )
		return predicate

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

	def extract_arguments(self, dependency_tree: Doc, min_arguments=0, using_default=False, arguments_predictor=None,
						  specify_none=False, trim_arguments=True, verb_noun_matcher=None, limited_verbs=None, predicate_indexes=None):
		"""
		Extracts the arguments for any relevant word of the given sentence that appear in this lexicon
		:param dependency_tree: the appropriate dependency tree for a sentence
		:param min_arguments: the minimum number of arguments for any founed extraction (0 is deafult)
		:param using_default: whether to use the default entry in the lexicon all of the time, otherwise only whenever it is needed
		:param arguments_predictor: the model-based extractor object to determine the argument type of a span (optional)
		:param specify_none: whether to specify in the resulted extractions about the unused arguments
		:param trim_arguments: whether to trim the argument spans in the resulted extractions
		:param verb_noun_matcher: a dictionary object of all the known nominalizations (optional)
		:param limited_verbs: a list of limited verbs, which limits the predicates that their arguments will be extracted (optional)
		:param predicate_indexes: a list of specific indexes of the predicated that should be extracted
		:return: all the founded argument extractions for any relevant word ({Token: [{COMP: Span}]})
		"""

		extractions_per_word = defaultdict(list)

		for token in dependency_tree:
			if predicate_indexes and token.i not in predicate_indexes:
				continue

			# Try to find an appropriate entry for the current word token
			word_entry, suitable_verb = self.find(token, using_default, verb_noun_matcher, limited_verbs)
			if not suitable_verb:
				continue

			# Get the candidates for the arguments of this word (based on relevant direct links in the ud dependency tree)
			argument_candidates = get_argument_candidates(token, include_nom=not self.is_verb and not arguments_predictor)
			if len(argument_candidates) > self.MAX_N_CANDIDATES:
				continue

			# The word itself can also be an argument
			# predicate = self._predicate_wrapper(token, word_entry, arguments_predictor)

			# if config.DEBUG:
				# print(f"Candidates for {token.orth_}:", [candidate_token._.subtree_text if candidate_token != token else candidate_token.orth_ for candidate_token in argument_candidates])

			# Get all the possible extractions of this word
			extractions = word_entry.match_arguments(argument_candidates, token, suitable_verb, arguments_predictor)

			for extraction in extractions:
				if len(extraction.get_complements()) >= min_arguments:
					# extraction.add_argument(ExtractedArgument())
					extraction_dict = extraction.as_span_dict(trim_arguments)
					self._update_unused_candidates(dependency_tree, argument_candidates, token, extraction.get_tokens(), extraction_dict, specify_none)
					extractions_per_word[token].append(extraction_dict)

			if config.DEBUG and len(extractions_per_word.get(token, [])) > 1:
				print(extractions_per_word[token])
				raise Exception("Found word with more than one legal extraction.")

		return extractions_per_word