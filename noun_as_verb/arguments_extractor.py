import os
import re
from collections import defaultdict

from tqdm import tqdm

from noun_as_verb.rule_based import Lexicon
from noun_as_verb.lisp_to_json import lisp_to_json
from noun_as_verb.model_based.types_predictor import TypesPredictor
from noun_as_verb.verb_noun_matcher import VerbNounMatcher
from noun_as_verb.constants.ud_constants import *
from noun_as_verb.utils import get_lexicon_path, get_dependency_tree, flatten, filter_list
from noun_as_verb import config

class ArgumentsExtractor:
	PREDICATE = "#"

	verb_lexicon: Lexicon
	nom_lexicon: Lexicon
	transer_args_predictor: TypesPredictor
	context_args_predictor: TypesPredictor
	verb_noun_matcher: VerbNounMatcher

	def __init__(self, lexicon_file_name=config.LEXICON_FILE_NAME):
		verb_json_file_path = get_lexicon_path(lexicon_file_name, "json", is_verb=True)
		nom_json_file_path = get_lexicon_path(lexicon_file_name, "json", is_nom=True)

		# Should we create the JSON formated lexicon again?
		if not (config.LOAD_LEXICON and os.path.exists(verb_json_file_path) and os.path.exists(nom_json_file_path)):
			lisp_to_json(lexicon_file_name)

		# Create the lexicon objects
		self.verb_lexicon = Lexicon(lexicon_file_name, is_verb=True)
		self.nom_lexicon = Lexicon(lexicon_file_name, is_verb=False)

		# Load the predicator of arguments (without context)
		self.transer_args_predictor = TypesPredictor({"SUBJECT", "OBJECT", "NONE"})
		self.transer_args_predictor.load_model()

		# Load the predictor of the arguments (with context)
		#self.context_args_predictor = TypesPredictor(SYN_ARGS_LABELS)
		#self.context_args_predictor.load_model()

		# Load the dictionary of all possible known nominalization
		self.verb_noun_matcher = VerbNounMatcher(self.verb_lexicon, self.nom_lexicon)



	def extractions_as_mentions(self, extractions_per_predicate, document_id, sentence_id, first_word_index, left_shift=0):
		mentions = []
		first_word_index += left_shift

		event_id = 0
		argument_id = 0

		predicate_tokens = list(extractions_per_predicate.keys())
		predicate_tokens.sort(key=lambda token: token.i)

		for predicate_token in predicate_tokens:
			extractions = extractions_per_predicate[predicate_token]
			sentence_tokens = predicate_token.doc

			# Check whether the current extractions are of a verb or a nominalization
			is_verb_related = predicate_token.pos_ == UPOS_VERB
			suitable_verb = predicate_token.lemma_

			if not is_verb_related:
				_, suitable_verb = self.nom_lexicon.find(predicate_token, using_default=True, verb_noun_matcher=self.verb_noun_matcher)

			if len(extractions) > 1:
				suitable_verb += f" ({len(extractions)})"

			all_argument_indexes = [left_shift + predicate_token.i]
			arguments_dict = defaultdict(list)

			if extractions != []:
				extraction = extractions[0]

				# Create a text mention for each argument of the predicate
				for argument_type, argument_span in extraction.items():
					start_index = left_shift + argument_span[0].i
					end_index = left_shift + argument_span[-1].i

					argument_mention = {
						"type": "TextBoundMention",
						"id": f"T:{sentence_id},{event_id},{argument_id}",
						"text": argument_span.orth_,
						"labels": ["\xa0"],
						"tokenInterval": {
							"start": start_index,
							"end": end_index + 1
						},
						"sentence": sentence_id,
						"document": document_id,
						"event": event_id,
						"isVerbRelated": is_verb_related
					}

					all_argument_indexes += [start_index, end_index]
					mentions.append(argument_mention)
					argument_id += 1

					arguments_dict[argument_type].append(argument_mention)

			start_index = min(all_argument_indexes)
			end_index = max(all_argument_indexes)

			# Create an event mention for the predicate
			event_mention = {
				"type": "EventMention",
				"id": f"R:{sentence_id},{event_id}",
				"text": sentence_tokens[start_index: end_index + 1].orth_,
				"labels": [suitable_verb],
				"sentence": sentence_id,
				"document": document_id,
				"event": event_id,

				"trigger": {
					"type": "TextBoundMention",
					"id": f"T:{sentence_id},{event_id},{argument_id}",
					"text": predicate_token.orth_,
					"labels": ["\xa0"],
					"tokenInterval": {
						"start": left_shift + predicate_token.i,
						"end": left_shift + predicate_token.i + 1
					},
					"realIndex": first_word_index + predicate_token.i,
					"sentence": sentence_id,
					"document": document_id
				},

				"arguments": arguments_dict,
				"isVerbRelated": is_verb_related
			}

			# In case that the prediate do not have any possible extractions, than it be presented as a text mention
			if extractions == []:
				event_mention["type"] = "TextBoundMention"
				event_mention["id"] = f"T:{sentence_id},{event_id}"
				event_mention["tokenInterval"] = {
					"start": left_shift + predicate_token.i,
					"end": left_shift + predicate_token.i + 1
				}

			argument_id += 1
			event_id += 1
			mentions.append(event_mention)

		return mentions

	@staticmethod
	def extractions_as_IOB(extractions_per_predicate):
		iob_format_extractions = []

		for predicate_token, extractions in extractions_per_predicate.items():
			dependency_tree = predicate_token.doc
			iob_format_extraction = [[word, "O"] for word in dependency_tree]
			iob_format_extraction[predicate_token.i][1] = "VERB" if predicate_token.pos_ == UPOS_VERB else "NOM"

			if extractions != []:
				extraction = extractions[0]

				for complement_type, argument_span in extraction.items():
					for argument_token in argument_span:
						if argument_token == argument_span[0]:
							label = "B"
						else:
							label = "I"

						iob_format_extraction[argument_token.i][1] = label + "-" + complement_type

			iob_format_extraction = " ".join([f"{word}/{tag}" for word, tag in iob_format_extraction]) + "\n"

			iob_format_extractions.append(iob_format_extraction)

		return iob_format_extractions

	@staticmethod
	def extractions_as_IOBES(extractions_per_predicate):
		iobes_format_extractions = []

		for predicate_token, extractions in extractions_per_predicate.items():
			dependency_tree = predicate_token.doc

			# if extractions != []:
			# 	extraction = extractions[0]

			if extractions == []:
				continue

			extractions = [extractions[0]]

			for extraction in extractions:
				iobes_format_extraction = [[word, "O"] for word in dependency_tree]
				iobes_format_extraction[predicate_token.i][1] = "O-VERB" if predicate_token.pos_ == UPOS_VERB else "O-NOM"

				for complement_type, argument_span in extraction.items():
					for argument_token in argument_span:
						if len(argument_span) == 1:
							label = "S"
						elif argument_token == argument_span[0]:
							label = "B"
						elif argument_token == argument_span[-1]:
							label = "E"
						else:
							label = "I"

						if iobes_format_extraction[argument_token.i][1] != "O":
							iobes_format_extraction[argument_token.i][1] = iobes_format_extraction[argument_token.i][1].replace("O-", label + "-" + complement_type + "-")
						else:
							iobes_format_extraction[argument_token.i][1] = label + "-" + complement_type

				iobes_format_extraction = " ".join([f"{word}/{tag}" for word, tag in iobes_format_extraction]) + "\n"
				iobes_format_extractions.append(iobes_format_extraction)

		return iobes_format_extractions

	@staticmethod
	def extractions_as_str(extractions_per_predicate):
		str_extractions_per_predicate = {}

		for word_token, extractions in extractions_per_predicate.items():
			str_extractions = []

			for extraction in extractions:
				str_extraction = {}
				for complement_type, argument_span in extraction.items():
					str_extraction[complement_type] = argument_span.orth_

				str_extractions.append(str_extraction)

			str_extractions_per_predicate[word_token.orth_] = str_extractions

		return str_extractions_per_predicate


	def new_extract_arguments(self, sentence):
		dependency_tree = get_dependency_tree(sentence)

		for token in dependency_tree:
			word_entry = self.search_word()

			if not word_entry:
				continue

	def extract_arguments(self, sentence, return_dependency_tree=False, min_arguments=0, using_default=False, transer_args_predictor=None, context_args_predictor=None,
						  specify_none=False, trim_arguments=True, verb_noun_matcher=None, limited_verbs=None, predicate_indexes=None, return_single=False):
		"""
		Extracts arguments of nominalizations and verbs in the given sentence, using NOMLEX lexicon
		:param sentence: a string text or a dependency tree parsing of a sentence
		:param return_dependency_tree: whether to return the depenency tree of the given sentence as a third parameter (optional)
		:param min_arguments: the minimum number of arguments for any founed extraction (0 is deafult)
		:param using_default: whether to use the default entry in the lexicon all of the time, otherwise only whenever it is needed
		:param arguments_predictor: the model-based extractor object to determine the argument type of a span (optional)
		:param specify_none: whether to specify in the resulted extractions about the unused arguments
		:param trim_arguments: whether to trim the argument spans in the resulted extractions
		:param verb_noun_matcher: a object that matches a verb with nouns whitin its word family using CATVAR (optional)
		:param limited_verbs: a list of limited verbs, which limits the predicates that their arguments will be extracted (optional)
		:param predicate_indexes: a list of specific indexes of the predicated that should be extracted
		:param return_single: whether to return only a single dictionary of the extracted arguments, together
		:return: Two dictionaries (and an optional dependency tree):
				 - The founded extractions for each relevant verbs in the given sentence ({verb_Token: [extraction_Span]})
				 - The founded extractions for each relevant nominalizations in the given sentence ({nom_Token: [extraction_Span]})
				 The two dictionaries may return as a single dictionary
		"""

		dependency_tree = get_dependency_tree(sentence)

		# Extract arguments based on the verbal lexicon
		#extractions_per_verb = self.verb_lexicon.extract_arguments(dependency_tree, min_arguments, using_default, transer_args_predictor, context_args_predictor,
		#														   specify_none, trim_arguments, verb_noun_matcher, limited_verbs, predicate_indexes)
		extractions_per_verb = {}

		# Extract arguments based on the nominal lexicon
		extractions_per_nom = self.nom_lexicon.extract_arguments(dependency_tree, min_arguments, using_default, transer_args_predictor, context_args_predictor,
																 specify_none, trim_arguments, verb_noun_matcher, limited_verbs, predicate_indexes)

		if return_single:
			extractions_per_word = extractions_per_verb
			extractions_per_word.update(extractions_per_nom)

			if return_dependency_tree:
				return extractions_per_word, dependency_tree
			else:
				return extractions_per_word

		if return_dependency_tree:
			return extractions_per_verb, extractions_per_nom, dependency_tree
		else:
			return extractions_per_verb, extractions_per_nom

	def rule_based_extraction(self, sentence, return_dependency_tree=False, min_arguments=0, limited_verbs=None, predicate_indexes=None, return_single=False):
		return self.extract_arguments(sentence, return_dependency_tree, min_arguments, transer_args_predictor=self.transer_args_predictor,
									  limited_verbs=limited_verbs, predicate_indexes=predicate_indexes, return_single=return_single)

	def enhanced_rule_based_extraction(self, sentence, return_dependency_tree=False, min_arguments=0, limited_verbs=None, predicate_indexes=None, return_single=False):
		return self.extract_arguments(sentence, return_dependency_tree, min_arguments, transer_args_predictor=self.transer_args_predictor,
									  limited_verbs=limited_verbs, predicate_indexes=predicate_indexes, return_single=return_single)

	def hybrid_based_extraction(self, sentence, return_dependency_tree=False, min_arguments=0, limited_verbs=None, predicate_indexes=None, return_single=False):
		return self.extract_arguments(sentence, return_dependency_tree, min_arguments, transer_args_predictor=self.transer_args_predictor, context_args_predictor=self.context_args_predictor,
									  verb_noun_matcher=self.verb_noun_matcher, limited_verbs=limited_verbs, predicate_indexes=predicate_indexes, return_single=return_single)

	def model_based_extraction(self, sentence, return_dependency_tree=False, min_arguments=0, limited_verbs=None, predicate_indexes=None, return_single=False):
		return self.extract_arguments(sentence, return_dependency_tree, min_arguments, using_default=True, transer_args_predictor=self.transer_args_predictor, context_args_predictor=self.context_args_predictor,
									  verb_noun_matcher=self.verb_noun_matcher, limited_verbs=limited_verbs, predicate_indexes=predicate_indexes, return_single=return_single)



	def _extract_specified_info(self, sentence):
		# Extracts the specified information in the given sentence, including argument names and chosen predicats

		get_first_tag = lambda s: re.search(r"\[([^\[\]]*)\]", s)
		specified_arguments = {}
		predicate_indexes = []

		match = get_first_tag(sentence)
		while match:
			span = match.span()
			specified_value = match.group(1).split()

			# Save the founded tag
			if len(specified_value) != 1:
				tag = specified_value[0]
				span_indexes = (len(sentence[:span[0]].split()), len(sentence[:span[1]].split())-2)
				if tag == self.PREDICATE and span_indexes[0] == span_indexes[1]:
					predicate_indexes.append(span_indexes[0])
				else:
					specified_arguments[span_indexes] = tag

			# Remove the founded tag from the sentence
			sentence = str(sentence[0:span[0]] + " ".join(specified_value[1:]) + sentence[span[1]:])

			# Searching for another one
			match = get_first_tag(sentence)

		return sentence, predicate_indexes, specified_arguments

	def get_searched_args(self, example_sentence, extractor_function, specified_args=None, predicate_index=None):
		if type(example_sentence) != list:
			example_sentence = [example_sentence]
			specified_args = [specified_args]
			predicate_index = [predicate_index]

		searched_args_per_verb = {}
		triplets = zip(example_sentence, predicate_index, specified_args)
		for sent, predicate_index, specified_args in triplets:
			if specified_args is None or predicate_index is None:
				sent, predicate_indexes, specified_args = self._extract_specified_info(sent)

				if len(predicate_indexes) >= 1:
					predicate_index = predicate_indexes[0]

			if predicate_index is None:
				continue

			extractions_per_word = extractor_function(self, sent, predicate_indexes=[predicate_index], return_single=True)

			if extractions_per_word is {}:
				continue

			# At most only one word was found
			word = list(extractions_per_word.keys())[0]
			suitable_verb = self.verb_noun_matcher.get_suitable_verb(word)
			extractions = extractions_per_word[word]

			# Generate the arguments translator based on the arguments whithin the extractions
			searched_args = defaultdict(list)
			for arg_type, arg_span in flatten([list(extraction.items()) for extraction in extractions]):
				span_indexes = (arg_span[0].i, arg_span[-1].i)
				if span_indexes in specified_args:
					searched_args[arg_type] = specified_args[span_indexes]

			searched_args_per_verb[suitable_verb] = searched_args

		return searched_args_per_verb

	def _translate_extractions(self, extractions_per_word, searched_args):
		# Translates the given extractions, according to the chosen arguments
		trans_extractions_per_word = defaultdict(list)

		for predicate, extractions in extractions_per_word.items():
			suitable_verb = self.verb_noun_matcher.get_suitable_verb(predicate)
			trans_extractions = []

			if suitable_verb not in searched_args:
				continue

			# Translate the extraction
			for extraction in extractions:
				joint_args = set(searched_args[suitable_verb].keys()).intersection(extraction.keys())
				trans_extrct = {searched_args[suitable_verb][arg]: extraction[arg] for arg in joint_args}
				trans_extractions.append(trans_extrct)

			# Filter out repeated extractions
			filtered_extractions = filter_list(trans_extractions, lambda d1,d2: d1.items()<=d2.items(), lambda d: d.keys(), greedy=True)
			trans_extractions_per_word[predicate] = filtered_extractions

		return trans_extractions_per_word

	def search_matching_extractions(self, searched_args, sentences, extractor_function, limited_results=None):
		limited_words, limited_verbs = set(), set()
		for suitable_verb in searched_args.keys():
			limited_words.update(self.verb_noun_matcher.get_all_forms(suitable_verb))
			limited_verbs.add(suitable_verb)

		if limited_words == set():
			return {}

		# Extract the same arguments for every nominalization or verb in the given list of sentences
		count_sents = 0
		matching_extractions = defaultdict(list)
		for sentence in tqdm(sentences):
			doc = get_dependency_tree(sentence, disable=['ner', 'parser', 'tagger'])
			lemmas = [w.lemma_ for w in doc]

			# Does any of the searched words (verbs and noms) appear in the sentence?
			if limited_words.isdisjoint(lemmas):
				continue

			# Get the extractions of the sentence
			extractions_per_word = extractor_function(self, sentence, limited_verbs=list(limited_verbs), return_single=True)

			# Replace arguments with the searched arguments names
			trans_extractions_per_word = self._translate_extractions(extractions_per_word, searched_args)
			matching_extractions.update(trans_extractions_per_word)

			if len(trans_extractions_per_word) >= 1:
				count_sents += 1

			if limited_results and count_sents >= limited_results:
				break

		return matching_extractions

	def rule_based_search(self, searched_args, sentences, limited_results=None):
		return self.search_matching_extractions(searched_args, sentences, self.rule_based_extraction, limited_results=limited_results)

	def hybrid_based_search(self, searched_args, sentences, limited_results=None):
		return self.search_matching_extractions(searched_args, sentences, self.hybrid_based_extraction, limited_results=limited_results)

	def model_based_search(self, searched_args, sentences, limited_results=None):
		return self.search_matching_extractions(searched_args, sentences, self.model_based_extraction, limited_results=limited_results)