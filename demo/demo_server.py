import sys, os, traceback
from bottle import route, run, request, static_file
import spacy
from spacy.tokens import Doc
from pybart.api import Converter
from pybart.converter import ConvsCanceler
from collections import defaultdict
import pybart.conllu_wrapper as cw
from getpass import getpass
import ssl
import smtplib

#@TODO- change import of module when it will become a web package
sys.path.append("../")
from arguments_extractor import *
argumentExtractor = ArgumentsExtractor("NOMLEX-plus.1.0.txt")

def extractions_to_mentions(extractions_per_word, sentence_tokens, document_id, sentence_id, event_id, argument_id):
	mentions = []

	for word, extractions in extractions_per_word.items():
		if extractions == []:
			continue

		event_id += 1

		word_str, word_index = word
		extraction = extractions[0]
		all_argument_indexes = [word_index]
		arguments_dict = defaultdict(list)

		for argument, argument_indexes in extraction.items():
			start_index = min(argument_indexes)
			end_index = max(argument_indexes)

			new_mention = {
				"type": "TextBoundMention",
				"id": f"T:{argument_id}",
				"text": " ".join(sentence_tokens[start_index: end_index + 1]),
				"labels": ["\xa0"],
				"tokenInterval": {
					"start": start_index,
					"end": end_index + 1
				},
				"sentence": sentence_id,
				"document": document_id
			}

			all_argument_indexes += argument_indexes
			mentions.append(new_mention)
			argument_id += 1

			arguments_dict[argument].append(new_mention)

		start_index = min(all_argument_indexes)
		end_index = max(all_argument_indexes)

		new_mention = {
			"type": "EventMention",
			"id": f"E:{event_id}",
			"text": " ".join(sentence_tokens[start_index: end_index + 1]),
			"labels": [sentence_tokens[word_index]],
			"sentence": sentence_id,
			"document": document_id,

			"trigger": {
				"type": "TextBoundMention",
				"id": f"T:{argument_id}",
				"text": sentence_tokens[word_index],
				"labels": ["\xa0"],
				"tokenInterval": {
					"start": word_index,
					"end": word_index + 1
				},
				"sentence": sentence_id,
				"document": document_id
			},

			"arguments": arguments_dict
		}

		argument_id += 1
		mentions.append(new_mention)

	return mentions, event_id, argument_id

def get_list_of_files(dir_name):
	# create a list of file and sub directories
	# names in the given directory
	list_of_file = os.listdir(dir_name)
	all_files = list()
	# Iterate over all the entries
	for entry in list_of_file:
		# Create full path
		full_path = os.path.join(dir_name, entry)
		# If entry is a directory then get the list of files in this directory
		if os.path.isdir(full_path):
			all_files = all_files + get_list_of_files(full_path)
		else:
			all_files.append(full_path.replace("./", ""))

	return all_files

@route('/nomlexDemo/')
@route('/nomlexDemo/<file_path:path>')
def server_static(file_path="index.html"):
	last_part = file_path[file_path.rfind('/') + 1:]
	all_files = get_list_of_files(".")

	# Check whether the ending of the file is a new of a file, otherwise is it is an input text
	if (" " in last_part) or (file_path not in all_files):
		file_path = file_path.replace(last_part, "index.html")

	return static_file(file_path, root='./')


@route('/nomlexDemo/feedback/', method='POST')
def feedback():
	text_to_send = request.json["text_to_send"]
	port = 465  # For SSL
	smtp_server = "smtp.gmail.com"
	sender_email = "aviv.wn@gmail.com"
	receiver_email = "aviv.wn@gmail.com"
	message = 'Subject: {}\n\n{}'.format("Argument Extraction Feedback", text_to_send)
	context = ssl.create_default_context()
	try:
		with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
			print(1)
			server.login(sender_email, password)
			print(2)
			server.sendmail(sender_email, receiver_email, message)
	except smtplib.SMTPException:
		exc_type, exc_value, exc_traceback = sys.exc_info()
		traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
		with open("feedback.log", "a") as f:
			f.write(text_to_send + "\n")

@route('/nomlexDemo/annotate/', method='POST')
def annotate():
	sentence = request.json["sentence"]
	include_verbs = request.json["include_verbs"]
	include_noms = request.json["include_noms"]

	#sentence = "Paris's destruction"

	#doc = nlp(sentence)
	# doc = Doc(nlp.vocab, words=[t.text for t in nlp(sentence) if not t.is_space])
	# tagger(doc)
	# parser(doc)
	# basic_con(doc)
	# basic_con = Converter(False, False, False, 0, False, False, False, False, False, ConvsCanceler())
	# nlp.add_pipe(basic_con, name="BART")
	# doc = nlp(sentence)

	ud_doc = Doc(nlp.vocab, words=[t.text for t in nlp(sentence) if not t.is_space])

	converter = Converter(False, False, False, 0, False, False, False, False, False, ConvsCanceler())
	tagger(ud_doc)
	parser(ud_doc)
	converter(ud_doc)

	odin_formated_doc = cw.conllu_to_odin(converter.get_parsed_doc(), is_basic=True, push_new_to_end=False)
	document_id = ""
	sentence_id = 0
	event_id = 0
	argument_id = 0
	odin_formated_doc["mentions"] = []

	# Add the mentions
	for sentence in odin_formated_doc["documents"][""]["sentences"]:
		sentence_tokens = sentence["words"]
		sentence_text = " ".join(sentence_tokens)

		extractions_per_words = argumentExtractor.extract_arguments(sentence_text, as_indexes=True, include_verbs=include_verbs, include_noms=include_noms)
		mentions, event_id, argument_id = extractions_to_mentions(extractions_per_words, sentence_tokens, document_id, sentence_id, event_id, argument_id)
		odin_formated_doc["mentions"] += mentions
		sentence_id += 1

	return odin_formated_doc

	# basic_con = Converter(False, False, False, 0, False, False, False, False, False, ConvsCanceler())
	# doc = Doc(nlp.vocab, words=[t.text for t in nlp(sentence) if not t.is_space])
	# tagger(doc)
	# parser(doc)
	# basic_con(doc)
	# odin_basic_out = cw.conllu_to_odin(basic_con.get_parsed_doc(), is_basic=True, push_new_to_end=False)
	#displacy.parse_deps(doc)
	# print(odin_basic_out)
	#
	# return odin_basic_out

	#basic_doc = Doc(nlp.vocab, words=[t.text for t in nlp(sentence) if not t.is_space])
	#extra_doc = Doc(nlp.vocab, words=[t.text for t in nlp(sentence) if not t.is_space])

	# basic_con = Converter(False, False, False, 0, False, False, False, False, False, ConvsCanceler())
	# extra_con = Converter(eud, eud_pp, eud_bart, int(conv_iterations) if conv_iterations != "inf" else math.inf,
	#                       remove_eud_info,
	#                       not include_bart_info, remove_node_adding_convs, False, False, ConvsCanceler())
	#
	# for doc, con in [(basic_doc, basic_con), (extra_doc, extra_con)]:
	#     _ = tagger(doc)
	#     _ = parser(doc)
	#     _ = con(doc)
	#
	# odin_basic_out = cw.conllu_to_odin(basic_con.get_parsed_doc(), is_basic=True, push_new_to_end=False)
	# odin_plus_out = cw.conllu_to_odin(extra_con.get_parsed_doc(), push_new_to_end=False)
	#
	# return json.dumps({
	#     "basic": odin_basic_out,
	#     "plus": odin_plus_out,
	#     "conv_done": extra_con.get_max_convs(),
	# })


# password = input("password for sending emails: ")
password = getpass("Password for sending emails: ")
nlp = spacy.load("en_ud_model_lg")
tagger = nlp.get_pipe('tagger')
parser = nlp.get_pipe('parser')
run(host='0.0.0.0', reloader=False, port=5001)