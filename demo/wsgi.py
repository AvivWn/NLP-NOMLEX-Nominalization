import os
import sys
import traceback
import ssl
import smtplib
from random import shuffle
from uuid import uuid4
from typing import Tuple
from dataclasses import asdict

import numpy as np
from bottle import route, run, request, static_file

from yet_another_verb.arguments_extractor.extraction.comparators.arg_type_matcher import ArgTypeMatcher
from yet_another_verb.arguments_extractor.extraction.extraction import Extractions
from yet_another_verb.configuration.parsing_config import PARSING_CONFIG
from yet_another_verb import NomlexArgsExtractor
from demo.dynamic_extractions_info import DynamicExtractionsInfo
from demo.handle_specified_tags import parse_specified_tags, translate_char_ranges_to_word_ranges
from handle_args_extraction import generate_args_extraction_info
from time_utils import force_min_response_time
from config import DEMO_CONFIG


def _get_list_of_files(dir_name):
	# Creates a list of all the files in the given directory, recursively

	list_of_file = os.listdir(dir_name)
	all_files = list()

	for entry in list_of_file:
		full_path = os.path.join(dir_name, entry)

		if os.path.isdir(full_path):
			all_files = all_files + _get_list_of_files(full_path)
		else:
			all_files.append(full_path.replace("./", ""))

	return all_files


def _get_random_sentence(sentences):
	random_sentence = sentences[np.random.choice(len(sentences))]
	return random_sentence


@route('/nomlexDemo/')
@route('/nomlexDemo/<file_path:path>')
def server_static(file_path="index.html"):
	last_part = file_path[file_path.rfind('/') + 1:]
	all_files = _get_list_of_files(".")

	# Check whether the ending is a name of a file, otherwise it is an input text
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

	try:
		with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
			server.login(sender_email, password)
			server.sendmail(sender_email, receiver_email, message)

	# If the smtp servers fails, the feedback is written instead in a dedicated file
	except smtplib.SMTPException:
		exc_type, exc_value, exc_traceback = sys.exc_info()
		traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
		with open("feedback/feedback_log.txt", "a") as f:
			f.write(text_to_send + "\n")


@route('/nomlexDemo/get_random_example/', method='GET')
def get_random_example():
	return {"text": _get_random_sentence(example_sentences)}


def _parse_text(text: str):
	clean_text, tagged_ranges = parse_specified_tags(text)
	ud_parsed_text = ud_parser(clean_text)
	return ud_parsed_text, tagged_ranges


def _extract_inner(text: str, args_extractor) -> Tuple[Extractions, DynamicExtractionsInfo]:
	ud_parsed_text, tagged_ranges = _parse_text(text)
	tagged_ranges = translate_char_ranges_to_word_ranges(ud_parsed_text, tagged_ranges)
	return generate_args_extraction_info(ud_parsed_text, args_extractor, str(uuid4()), tagged_ranges)


@route('/nomlexDemo/extract/', method='POST')
@force_min_response_time
def extract():
	text = request.json["text"]
	extraction_based = request.json["extraction-based"]
	args_extractor = args_extractors[extraction_based]
	_, dynamic_extractions_info = _extract_inner(text, args_extractor)
	return asdict(dynamic_extractions_info)


@route('/nomlexDemo/match/', method='POST')
@force_min_response_time
def match():
	text = request.json["text"]
	extraction_based = request.json["extraction-based"]
	args_extractor = args_extractors[extraction_based]
	tagged_extractions, _ = _extract_inner(text, args_extractor)

	dynamic_extractions_info = DynamicExtractionsInfo()

	document_id = str(uuid4())
	shuffle(parsed_example_documents)
	for parsed_text in parsed_example_documents:
		_, dynamic_extractions_info = generate_args_extraction_info(
			parsed_text, args_extractor, document_id,
			references=tagged_extractions, reference_matcher=extraction_matcher,
			dynamic_extractions_info=dynamic_extractions_info
		)

		if len(dynamic_extractions_info.mentions_by_event) > DEMO_CONFIG.MAX_MATCHING_RESULTS:
			break

	return asdict(dynamic_extractions_info)


if __name__ == '__main__':
	# Email for feedback response
	email_address = "" #input("Enter your e-mail address: ")
	password = "" #getpass("Password for sending emails: ")

	ud_parser = PARSING_CONFIG.DEFAULT_PARSER_MAKER()
	default_args_extractor = NomlexArgsExtractor()
	args_extractors = {
		"rule-based": default_args_extractor
	}
	extraction_matcher = ArgTypeMatcher()

	with open(DEMO_CONFIG.EXAMPLE_DATA_PATH, "r") as example_sentence_file:
		example_sentences = example_sentence_file.readlines()

	with open(DEMO_CONFIG.PARSED_EXAMPLE_DATA_PATH, "rb") as parsed_dataset_file:
		dataset_bytes = parsed_dataset_file.read()
		doc_bin = PARSING_CONFIG.DEFAULT_PARSED_BIN_MAKER().from_bytes(dataset_bytes)
		parsed_example_documents = list(doc_bin.get_parsed_texts(default_args_extractor.dependency_parser))

	run(host=DEMO_CONFIG.URL, reloader=False, port=DEMO_CONFIG.PORT, server='paste')
