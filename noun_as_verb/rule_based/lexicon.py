import os
import json
import pickle
from collections import defaultdict
import itertools

from spacy.tokens import Token, Doc
from tqdm import tqdm

from noun_as_verb.lisp_to_json import lisp_to_json
from noun_as_verb.rule_based import Entry
from noun_as_verb.rule_based import get_argument_candidates
from noun_as_verb.rule_based import ExtractedArgument
from noun_as_verb.utils import get_lexicon_path, difference_list, filter_list, aggregate_to_dict
from noun_as_verb.constants.ud_constants import *
from noun_as_verb.constants.lexicon_constants import *
from noun_as_verb import config


class Lexicon:
	entries = defaultdict(Entry)
	is_verb: bool

	MAX_N_CANDIDATES = 10

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

			if not os.path.exists(json_file_path):
				lisp_to_json(lisp_file_name)

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

	def get_verbs_by_arg(self, complement_type):
		# Finds the verbs that can be complemented by the given complement type

		suitable_verbs = set()
		for entry in self.entries.values():
			if entry.is_default_entry():
				continue

			suitable_verb = entry.get_suitable_verb()

			# The complement type is the nom
			if complement_type in entry.get_nom_types(ignore_part=True):
				suitable_verbs.update([suitable_verb])

			# The argument of that type can complement the verb\nom
			for subcat in entry.get_subcats():
				if complement_type in subcat.get_args_types():
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

	def _wrap_predicate(self, word:Token, word_entry:Entry, arguments_predictor=None):
		noun_type = None
		entropy = None

		# Predicting the predicate type for noun
		if not self.is_verb and arguments_predictor and word_entry.is_default_entry():
			noun_type, entropy = arguments_predictor.determine_noun_type(word)

		predicate = ExtractedArgument(word, noun_type, matched_position=POS_NOM, entropy=entropy)
		return predicate

	def _update_unused_candidates(self, token_candidates: list, predicate_token: Token, used_tokens: list, extraction: dict, specify_none=False, trim_arguments=True):
		if not specify_none:
			return

		extraction[COMP_NONE] = []
		prepositions = list(itertools.chain.from_iterable([self.entries[DEFAULT_ENTRY].subcats[DEFAULT_SUBCAT].arguments[arg_type].prefixes for arg_type in
						[COMP_PP, COMP_IND_OBJ, COMP_SUBJ, COMP_OBJ]]))

		# Add any candidate that isn't in the used tokens to the NONE complement
		for unused_candidate in difference_list(token_candidates, used_tokens):
			unused_token = unused_candidate.get_token()

			nom_links = [URELATION_NMOD, URELATION_COMPOUND, URELATION_ACL]
			verb_links = [URELATION_NSUBJ, URELATION_IOBJ, URELATION_DOBJ, URELATION_NMOD_POSS, URELATION_NSUBJPASS, URELATION_NMOD]
			relevant_links = verb_links if self.is_verb else nom_links

			if unused_token.dep_ not in relevant_links or unused_token.i == predicate_token.i:
				continue

			if not unused_token.pos_.startswith("N"):
				continue

			if unused_token.dep_ in [URELATION_NMOD, URELATION_ACL]:
				found_prep = False
				candidate_text = unused_token._.subtree_text + " "

				for prefix in prepositions:
					if candidate_text.startswith(prefix):
						found_prep = True

				if not found_prep:
					continue

			unused_arg = ExtractedArgument(unused_token, COMP_NONE)
			arg_span = unused_arg.as_span(trim_argument=trim_arguments)
			extraction[COMP_NONE].append(arg_span)

	def _choose_informative(self, extractions: list, predicate: ExtractedArgument, suitable_verb: str, args_predictor=None):
		# Choose the extractions with max arguments
		all_extractions = extractions
		filter_extractions = lambda extractions: filter_list(extractions, lambda e1,e2: e2.is_sub_extraction(e1), lambda e: e.get_complements(), greedy=True)
		extractions = filter_extractions(extractions)

		# Choose only the informative extractions
		extractions = filter_list(extractions, lambda e1, e2: e1 != e2 and e1.is_more_informative(e2), lambda e: e.get_complements())

		# Uncertainty might already be handled by the other model (if the default entry was used)
		#if args_predictor and not self.is_default_entry():

		# Uncertainty should be handle if the args model is given
		if args_predictor and not self.is_verb:
			# Determine the complement type of uncertain candidates, with a model
			candidates_args = aggregate_to_dict([e.get_candidates_types() for e in extractions])
			candidates_args = args_predictor.determine_args_type(candidates_args, predicate, suitable_verb, default_subcat=True)

			# Clean extractions from arguments that weren't chosen
			extractions = [e for e in extractions if e.get_filtered(candidates_args).isin(all_extractions)]
			extractions = filter_extractions(extractions)

		return extractions

	def extract_arguments(self, dependency_tree: Doc, min_arguments=0, using_default=False, transer_args_predictor=None, context_args_predictor=None,
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
			argument_candidates = get_argument_candidates(token, include_nom=not self.is_verb and not context_args_predictor)
			#assert len(argument_candidates) <= self.MAX_N_CANDIDATES, (dependency_tree, token)
			if len(argument_candidates) > self.MAX_N_CANDIDATES:
				continue

			# The word itself can also be an argument
			predicate = self._wrap_predicate(token, word_entry, context_args_predictor)

			# if config.DEBUG:
				# print(f"Candidates for {token.orth_}:", [candidate_token._.subtree_text if candidate_token != token else candidate_token.orth_ for candidate_token in argument_candidates])

			# Get all the possible extractions of this word
			#get_extractions = timeit(word_entry.match_arguments)
			extractions = word_entry.match_arguments(argument_candidates, predicate, suitable_verb, context_args_predictor)

			# Choose the most informative extractions
			extractions = self._choose_informative(extractions, predicate, suitable_verb, transer_args_predictor)

			for extraction in extractions:
				if len(extraction.get_complements()) >= min_arguments:
					extraction_dict = extraction.as_span_dict(trim_arguments)

					if extraction_dict == {}:
						continue

					self._update_unused_candidates(argument_candidates, token, extraction.get_tokens(), extraction_dict, specify_none, trim_arguments)
					extractions_per_word[token].append(extraction_dict)

			if config.DEBUG and len(extractions_per_word.get(token, [])) > 1:
				pass
				# print(extractions_per_word[token])
				# raise Exception("Found word with more than one legal extraction.")

		return extractions_per_word