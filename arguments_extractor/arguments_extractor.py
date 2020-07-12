import os
from collections import defaultdict

from arguments_extractor.rule_based.lexicon import Lexicon
from arguments_extractor.lisp_to_json.lisp_to_json import lisp_to_json
from arguments_extractor.constants.ud_constants import *
from arguments_extractor.utils import get_lexicon_path, get_dependency_tree
from arguments_extractor import config

class ArgumentsExtractor:
	verb_lexicon: Lexicon
	nom_lexicon: Lexicon

	def __init__(self, lexicon_file_name):
		verb_json_file_path = get_lexicon_path(lexicon_file_name, "json", is_verb=True)
		nom_json_file_path = get_lexicon_path(lexicon_file_name, "json", is_nom=True)

		# Should we create the JSON formated lexicon again?
		if not (config.LOAD_LEXICON and os.path.exists(verb_json_file_path) and os.path.exists(nom_json_file_path)):
			lisp_to_json(lexicon_file_name)

		self.verb_lexicon = Lexicon(lexicon_file_name, is_verb=True)
		self.nom_lexicon = Lexicon(lexicon_file_name, is_verb=False)

	def get_n_predicates(self, word_tokens: list):
		n_predicates = 0

		for word in word_tokens:
			if word.lower() in self.verb_lexicon.entries.keys() or word.lower() in self.nom_lexicon.entries.keys():
				n_predicates += 1

		return n_predicates

	def split_noms(self, ratio):
		return self.nom_lexicon.split_noms(ratio)



	@staticmethod
	def extractions_as_mentions(extractions_per_predicate, document_id, sentence_id, first_word_index):
		mentions = []

		event_id = 0
		argument_id = 0

		predicate_tokens = list(extractions_per_predicate.keys())
		predicate_tokens.sort(key=lambda token: token.i)

		for predicate_token in predicate_tokens:
			extractions = extractions_per_predicate[predicate_token]
			sentence_tokens = predicate_token.doc

			# Check whether the current extractions are of a verb or a nominalization
			is_verb_related = predicate_token.pos_ == UPOS_VERB

			all_argument_indexes = [predicate_token.i]
			arguments_dict = defaultdict(list)

			if extractions != []:
				extraction = extractions[0]

				# Create a text mention for each argument of the predicate
				for argument_type, argument_span in extraction.items():
					start_index = argument_span[0].i
					end_index = argument_span[-1].i

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
				"labels": [predicate_token.orth_],
				"sentence": sentence_id,
				"document": document_id,
				"event": event_id,

				"trigger": {
					"type": "TextBoundMention",
					"id": f"T:{sentence_id},{event_id},{argument_id}",
					"text": predicate_token.orth_,
					"labels": ["\xa0"],
					"tokenInterval": {
						"start": predicate_token.i,
						"end": predicate_token.i + 1
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
					"start": predicate_token.i,
					"end": predicate_token.i + 1
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
			iobes_format_extraction = [[word, "O"] for word in dependency_tree]
			iobes_format_extraction[predicate_token.i][1] = "VERB" if predicate_token.pos_ == UPOS_VERB else "NOM"

			if extractions != []:
				extraction = extractions[0]

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



	def rule_based_extraction(self, sentence, return_dependency_tree=False, min_arguments=0):
		"""
		Extracts arguments of nominalizations and verbs in the given sentence, using NOMLEX lexicon
		:param sentence: a string text or a dependency tree parsing of a sentence
		:param return_dependency_tree: whether to return the depenency tree of the given sentence as a third parameter (optional)
		:param min_arguments: The minimum number of arguments for any founed extraction (0 is deafult)
		:return: Two dictionaries (and an optional dependency tree):
					The founded extractions for each relevant verbs in the given sentence ({verb_Token: [extraction_Span]})
					The founded extractions for each relevant nominalizations in the given sentence ({nom_Token: [extraction_Span]})
		"""

		if type(sentence) == str:
			dependency_tree = get_dependency_tree(sentence)
		else:
			dependency_tree = sentence

		# Extract arguments based on the verbal lexicon
		extractions_per_verb = self.verb_lexicon.extract_arguments(dependency_tree, min_arguments=min_arguments)

		# Extract arguments based on the nominal lexicon
		extractions_per_nom = self.nom_lexicon.extract_arguments(dependency_tree, min_arguments=min_arguments)

		if return_dependency_tree:
			return extractions_per_verb, extractions_per_nom, dependency_tree
		else:
			return extractions_per_verb, extractions_per_nom