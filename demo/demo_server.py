import os
import sys
import traceback
import ssl
import smtplib
import numpy as np
import time
from getpass import getpass
from random import shuffle
from collections import defaultdict
from bottle import route, run, request, static_file

import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
from spacy.tokens import DocBin
from pybart.api import Converter
from pybart.converter import ConvsCanceler
import pybart.conllu_wrapper as cw

#@TODO- change import of module when it will become a web package
sys.path.append("../")
from noun_as_verb import *

MIN_RESPONSE_TIME = 1 # seconds

def _custom_tokenizer(nlp):
	inf = list(nlp.Defaults.infixes)               # Default infixes
	inf.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")    # Remove the generic op between numbers or between a number and a -
	inf = tuple(inf)                               # Convert inf to tuple
	infixes = inf + tuple([r"(?<=[0-9])[+*^](?=[0-9-])", r"(?<=[0-9])-(?=-)"])  # Add the removed rule after subtracting (?<=[0-9])-(?=[0-9]) pattern
	infixes = [x for x in infixes if '-|–|—|--|---|——|~' not in x] # Remove - between letters rule
	infix_re = compile_infix_regex(infixes)

	return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
					 			suffix_search=nlp.tokenizer.suffix_search,
								infix_finditer=infix_re.finditer,
								token_match=nlp.tokenizer.token_match,
								rules=nlp.Defaults.tokenizer_exceptions)

def _get_list_of_files(dir_name):
	# Creates a list of all the files in the given directory, recursively

	# names in the given directory
	list_of_file = os.listdir(dir_name)
	all_files = list()
	# Iterate over all the entries
	for entry in list_of_file:
		# Create full path
		full_path = os.path.join(dir_name, entry)

		# If entry is a directory then get the list of files in this directory
		if os.path.isdir(full_path):
			all_files = all_files + _get_list_of_files(full_path)
		else:
			all_files.append(full_path.replace("./", ""))

	return all_files

def _get_random_sentence(sentences):
	random_sentence = sentences[np.random.choice(len(sentences))]
	return random_sentence

def _update_mentions_info(mentions, sentence_id, informative_events, mentions_by_events, event_by_word_index):
	mentions_by_events[sentence_id] = defaultdict(list)
	most_informative_events = {"verb": -1, "nom": -1}
	max_num_of_arguments = {"verb": -1, "nom": -1}

	for mention in mentions:
		is_verb_related = mention["isVerbRelated"]

		# Search for the event with the max number of arguments in the sentence (= the most informative event)
		if "arguments" in mention:
			event_type = "verb" if is_verb_related else "nom"

			num_of_arguments = len(mention["arguments"].keys())
			if num_of_arguments > max_num_of_arguments[event_type]:
				max_num_of_arguments[event_type] = num_of_arguments
				most_informative_events[event_type] = mention["event"]

		event_id = mention["event"]

		# The trigger mention (=event) will always appear first in the mentions list
		if "trigger" in mention:
			mentions_by_events[sentence_id][event_id] = [mention] + mentions_by_events[sentence_id][event_id]
			event_by_word_index[mention["trigger"]["realIndex"]] = {"event-id": event_id, "sentence-id": sentence_id, "is-verb": is_verb_related}
		else:
			mentions_by_events[sentence_id][event_id].append(mention)

	informative_events[sentence_id] = most_informative_events

def _iterator_in_range(iterator, range_start, range_end):
	if type(iterator) == list:
		new_iterator = []
	elif type(iterator) == dict:
		new_iterator = {}
	else:
		return None

	for x in iterator:
		start_index = x
		end_index = x

		if type(x) == tuple:
			start_index = x[0]
			end_index = x[1]
		elif type(x) != int:
			continue

		if start_index < range_start or end_index > range_end:
			continue

		if type(iterator) == list:
			new_iterator.append(x)
		elif type(iterator) == dict:
			new_iterator[x] = iterator[x]

	return new_iterator

def _extract_arguments(sentence, extractor_function, specified_arguments, predicate_indexes):
	sentence = sentence.strip(" \n\r\t")
	ud_doc = nlp(sentence)
	odin_formated_doc = cw.conllu_to_odin(converter.get_parsed_doc(), is_basic=True, push_new_to_end=False)

	document_id = ""
	sentence_id = 0
	first_word_index = 0
	docs = list(ud_doc.sents)

	mentions_by_events = {}  # list of mentions per event-id per sentence-id
	informative_events = {}  # event-id per sentence-id (the most informative events by the number of arguments)
	event_by_word_index = {}  # event-id, sentence-id, is-verb per word-id in the input string (that contains all the sentences together)

	extractions_per_word_list = []
	relevant_specified_arguments_list = []
	predicate_index_list = []

	# Generate the mentions for each sentence
	# The mentions will be based on the founded extractions (if any) in the sentences
	for i, sentence_info in enumerate(odin_formated_doc["documents"][""]["sentences"]):
		sentence_words = sentence_info["words"]
		sentence_doc = docs[i][:].as_doc()

		# Find the relevant tagged predicates and tagged arguments for this sentence
		relevant_predicate_indexes = _iterator_in_range(predicate_indexes, first_word_index, first_word_index + len(sentence_words))
		relevant_specified_arguments = _iterator_in_range(specified_arguments, first_word_index, first_word_index + len(sentence_words))

		# Extract arguments
		extractions_per_verb, extractions_per_nom, dependency_tree = extractor_function(test_extractor, sentence_doc, return_dependency_tree=True)
		postags = [token.pos_ for token in dependency_tree]
		sentence_info["tags"] = postags

		# Generate mentions
		extractions_per_word = extractions_per_verb
		extractions_per_word.update(extractions_per_nom)
		mentions = test_extractor.extractions_as_mentions(extractions_per_word, document_id, sentence_id, first_word_index)
		_update_mentions_info(mentions, sentence_id, informative_events, mentions_by_events, event_by_word_index)

		first_word_index += len(sentence_words)
		sentence_id += 1

		if relevant_predicate_indexes is None or relevant_predicate_indexes == []:
			continue

		extractions_per_word_list.append(extractions_per_word)
		relevant_specified_arguments_list.append(relevant_specified_arguments)
		predicate_index_list.append(relevant_predicate_indexes[0])

	# Find the total arguments that we want to search for each verb, and their updated names
	searched_args = test_extractor.get_searched_args(extractions_per_word_list, extractor_function,
													 specified_args=relevant_specified_arguments_list, predicate_index=predicate_index_list)

	extractions_info = {
		"parsed-data": odin_formated_doc,
		"mentions-by-events": mentions_by_events,
		"informative-events": informative_events,
		"event-by-word-index": event_by_word_index,
		"searched-args": searched_args
	}

	return extractions_info

def _search_matching_extractions(searched_args, parsed_example_sentences, extractor_function):
	mentions_by_events = {}  # list of mentions per event-id per sentence-id
	informative_events = {}  # event-id per sentence-id (the most informative events)
	event_by_word_index = {}  # event-id, sentence-id, is-verb per word-id in the input

	matching_extractions = test_extractor.search_matching_extractions(searched_args, parsed_example_sentences, extractor_function, limited_results=5)
	extractions_per_doc = defaultdict(dict)
	for predicate, extractions in matching_extractions.items():
		doc = predicate.doc
		extractions_per_doc[doc].update({predicate: extractions})

	document_id = ""
	sentence_id = 0
	first_word_index = 0

	odin_formated_doc = {"documents": {"": {"id": "", "text": "", "sentences": []}}, "mentions": []}

	for doc, extractions_per_word in extractions_per_doc.items():
		sentence_words = [str(sentence_id + 1) + ")"] + [t.orth_ for t in doc] + ["."]
		odin_formated_doc["documents"][""]["sentences"].append({"words": sentence_words, "tags": [], "graphs": {}})

		mentions = test_extractor.extractions_as_mentions(extractions_per_word, document_id, sentence_id, first_word_index, left_shift=1)
		_update_mentions_info(mentions, sentence_id, informative_events, mentions_by_events, event_by_word_index)

		first_word_index += len(sentence_words)
		sentence_id += 1

	matches_info = {
		"parsed-data": odin_formated_doc,
		"mentions-by-events": mentions_by_events,
		"informative-events": informative_events,
		"event-by-word-index": event_by_word_index,
	}

	return matches_info

def _choose_extractor_func(extraction_based):
	if extraction_based == "rule-based":
		extractor_function = ArgumentsExtractor.rule_based_extraction
	elif extraction_based == "model-based":
		extractor_function = ArgumentsExtractor.model_based_extraction
	else: # hybrid-based
		extractor_function = ArgumentsExtractor.hybrid_based_extraction

	return extractor_function



@route('/nomlexDemo/')
@route('/nomlexDemo/<file_path:path>')
def server_static(file_path="index.html"):
	last_part = file_path[file_path.rfind('/') + 1:]
	all_files = _get_list_of_files(".")

	# Check whether the ending of the file is a new of a file, otherwise is it is an input text
	if (" " in last_part) or (file_path not in all_files):
		file_path = file_path.replace(last_part, "index.html")

	return static_file(file_path, root='./')

@route('/nomlexDemo/feedback/', method='POST')
def feedback():
	text_to_send = request.json["text-to-send"]
	port = 465  # For SSL
	smtp_server = "smtp.gmail.com"
	sender_email = email_address
	receiver_email = email_address
	message = 'Subject: {}\n\n{}'.format("Argument Extraction Feedback", text_to_send)
	context = ssl.create_default_context()

	# Try to send the feedback e-mail
	try:
		with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
			server.login(sender_email, password)
			server.sendmail(sender_email, receiver_email, message)

	# If the smtp servers fails, the feedback is written instead in a dedicated file
	except smtplib.SMTPException:
		exc_type, exc_value, exc_traceback = sys.exc_info()
		traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
		with open("feedback-log.text", "a") as f:
			f.write(text_to_send + "\n")

@route('/nomlexDemo/extract/', method='POST')
def extract():
	start_time = time.time()

	original_sentence = request.json["sentence"]
	is_random = request.json["random"]
	extraction_based = request.json["extraction-based"]

	# Choose random sentence, if needed
	if is_random: original_sentence = _get_random_sentence(example_sentences)

	# Find all the specified arguments in the given sentence
	clean_sentence, predicate_indexes, specified_arguments = test_extractor._extract_specified_info(original_sentence)

	# Choose the appropriate extracting function
	extractor_function = _choose_extractor_func(extraction_based)

	# Extract the arguments
	extractions_info = _extract_arguments(clean_sentence, extractor_function, specified_arguments, predicate_indexes)

	if time.time() - start_time < MIN_RESPONSE_TIME:
		time.sleep(MIN_RESPONSE_TIME - (time.time() - start_time))

	return {"sentence": original_sentence, "extractions": extractions_info}

@route('/nomlexDemo/match/', method='POST')
def match():
	start_time = time.time()

	extraction_based = request.json["extraction-based"]
	extractions_info = request.json["extractions"]

	# Choose the appropriate extracting function
	extractor_function = _choose_extractor_func(extraction_based)

	# Search for matching extractions in the example sentences
	if extractions_info["searched-args"] == {}:
		matches_info = f"[THING No relevant arguments and predicates] were [{test_extractor.PREDICATE} specified] by [AGENT the user]."
	else:
		shuffle(parsed_example_sentences)
		matches_info = _search_matching_extractions(extractions_info["searched-args"], parsed_example_sentences, extractor_function)

		if matches_info["informative-events"] == {}:
			matches_info = "No matching extractions found!"

	if time.time() - start_time < MIN_RESPONSE_TIME:
		time.sleep(MIN_RESPONSE_TIME - (time.time() - start_time))

	return {"matches": matches_info}



if __name__ == '__main__':
	# Ask for password for the email respones
	email_address = input("Enter your e-mail address: ")
	password = getpass("Password for sending emails: ")

	# Create the arguments extractor
	test_extractor = ArgumentsExtractor("NOMLEX-plus.1.0.txt")

	# Create the UD parser, that resulted in odin formated representation
	nlp = spacy.load("en_ud_model_lg")
	nlp.tokenizer = _custom_tokenizer(nlp)
	converter = Converter(False, False, False, 0, False, False, False, False, False, ConvsCanceler())
	nlp.add_pipe(converter, name="BART")
	tagger = nlp.get_pipe('tagger')
	parser = nlp.get_pipe('parser')

	# Load the example sentences
	DATA_PATH = "data/too_clean_wiki/example.txt"
	with open(DATA_PATH, "r") as example_sentence_file:
		example_sentences = example_sentence_file.readlines()

	# Load the parsed example sentences
	with open(DATA_PATH.replace(".txt", ".parsed"), "rb") as parsed_dataset_file:
		dataset_bytes = parsed_dataset_file.read()
		doc_bin = DocBin().from_bytes(dataset_bytes)
		docs = doc_bin.get_docs(nlp.vocab)
		parsed_example_sentences = list(doc_bin.get_docs(nlp.vocab))

	# Start the server
	run(host='0.0.0.0', reloader=False, port=5000, server='paste')