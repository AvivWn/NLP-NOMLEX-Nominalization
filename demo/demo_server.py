import os
import sys
import traceback
import ssl
import smtplib
import numpy as np
from getpass import getpass
from collections import defaultdict
from bottle import route, run, request, static_file

import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
from pybart.api import Converter
from pybart.converter import ConvsCanceler
import pybart.conllu_wrapper as cw

#@TODO- change import of module when it will become a web package
sys.path.append("../")
from arguments_extractor import *
argumentExtractor = ArgumentsExtractor("NOMLEX-plus.1.0.txt")

DATA_PATH = "data/example_sentences.txt"
with open(DATA_PATH, "r") as sentences_file:
	sentences = sentences_file.readlines()

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

@route('/nomlexDemo/annotate/', method='POST')
def annotate():
	sentence = request.json["sentence"]
	is_random = request.json["random"]
	extraction_based = request.json["extraction-based"]

	if is_random:
		sentence = _get_random_sentence(sentences)

	sentence = sentence.strip()
	ud_doc = nlp(sentence)
	odin_formated_doc = cw.conllu_to_odin(converter.get_parsed_doc(), is_basic=True, push_new_to_end=False)

	document_id = ""
	sentence_id = 0
	first_word_index = 0

	mentions_by_events = {} # list of mentions per event-id per sentence-id
	informative_events = {} # event-id per sentence-id (the most informative events)
	event_by_word_index = {} # event-id, sentence-id, is-verb per word-id in the input

	sentence_docs = list(ud_doc.sents)

	# Generate the mentions for each sentence
	# The mentions will be based on the founded extractions (if any) in the sentences
	for i, sentence_info in enumerate(odin_formated_doc["documents"][""]["sentences"]):
		sentence_words = sentence_info["words"]
		sentence_text = " ".join(sentence_words)
		sentence_doc = sentence_docs[i][:].as_doc()

		if extraction_based == "rule-based":
			extractions_per_verb, extractions_per_nom, dependency_tree = argumentExtractor.rule_based_extraction(sentence_doc, return_dependency_tree=True)
		elif extraction_based == "model-based":
			extractions_per_verb, extractions_per_nom, dependency_tree = argumentExtractor.model_based_extraction(sentence_doc, return_dependency_tree=True)
		else: # hybrid-based
			extractions_per_verb, extractions_per_nom, dependency_tree = argumentExtractor.hybrid_based_extraction(sentence_doc, return_dependency_tree=True)

		postags = [token.pos_ for token in dependency_tree]
		sentence_info["tags"] = postags

		extractions_per_word = extractions_per_verb
		extractions_per_word.update(extractions_per_nom)
		mentions = argumentExtractor.extractions_as_mentions(extractions_per_word, document_id, sentence_id, first_word_index)
		_update_mentions_info(mentions, sentence_id, informative_events, mentions_by_events, event_by_word_index)

		first_word_index += len(sentence_words)
		sentence_id += 1

	return {
		"sentence": sentence,
		"parsed-data": odin_formated_doc,
		"mentions-by-events": mentions_by_events,
		"informative-events": informative_events,
		"event-by-word-index": event_by_word_index
	}



# Ask for password for the email respones
email_address = input("Enter your e-mail address: ")
password = getpass("Password for sending emails: ")

# Create the UD parser, that resulted in odin formated representation
nlp = spacy.load("en_ud_model_lg")

nlp.tokenizer = _custom_tokenizer(nlp)
converter = Converter(False, False, False, 0, False, False, False, False, False, ConvsCanceler())
nlp.add_pipe(converter, name="BART")
tagger = nlp.get_pipe('tagger')
parser = nlp.get_pipe('parser')
# annotate()

# Start the server
run(host='0.0.0.0', reloader=False, port=5000)