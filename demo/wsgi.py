import os
import sys
import traceback
import ssl
import smtplib
from random import shuffle
from uuid import uuid4
from typing import Tuple, List, Optional
from dataclasses import asdict

import numpy as np
from bottle import route, run, request, static_file

from yet_another_verb import NomlexArgsExtractor, ArgsExtractor
from yet_another_verb.arguments_extractor.extraction import Extractions, MultiWordExtraction
from demo.dynamic_extractions_info import DynamicExtractionsInfo
from demo.handle_specified_tags import parse_specified_tags, translate_char_ranges_to_word_ranges
from handle_args_extraction import generate_args_extraction_info
from time_utils import force_min_response_time
from config import DEMO_CONFIG
from yet_another_verb.arguments_extractor.extraction.utils.extraction_utils import reduce_extractions_by_arg_types
from yet_another_verb.arguments_extractor.extraction.utils.reconstruction import reconstruct_extraction
from yet_another_verb.arguments_extractor.extractors.verb_references_based.verb_references.utils import \
	get_references_by_predicate, load_extraction_references, get_closest_references, ScoredReference
from yet_another_verb.configuration.encoding_config import EncodingLevel, ENCODING_CONFIG
from yet_another_verb.configuration.extractors_config import EXTRACTORS_CONFIG, ExtractorType
from yet_another_verb.configuration.parsing_config import PARSING_CONFIG
from yet_another_verb.dependency_parsing import POSTag
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.factories.argument_encoder_factory import ArgumentEncoderFactory
from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory
from yet_another_verb.data_handling import TXTFileHandler, ExtractedFileHandler
from yet_another_verb.factories.extractor_factory import ExtractorFactory
from yet_another_verb.factories.verb_translator_factory import VerbTranslatorFactory
from yet_another_verb.nomlex.nomlex_version import NomlexVersion
from yet_another_verb.sentence_encoding.argument_encoding.utils import arg_encoder_to_tuple_id


def _get_examples():
	return TXTFileHandler(as_lines=True).load(DEMO_CONFIG.EXAMPLE_DATA_PATH)


def _get_arg_extractors():
	_args_extractors = {
		ExtractorType.NOMLEX: NomlexArgsExtractor(nomlex_version, ud_parser)
	}

	for extractor_type in [ExtractorType.DEPENDENCY_AVG_ARG, ExtractorType.DEPENDENCY_ARG_KNN]:
		extractor = ExtractorFactory(
			extractor_type,
			dependency_parser=ud_parser,
			verb_translator=verb_translator,
			arg_encoder=arg_encoder,
			references_by_verb=references_by_verb
		)()
		_args_extractors[extractor_type] = extractor

	return _args_extractors


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


@route(f'/{DEMO_CONFIG.WWW_ENDPOINT}/')
@route(f'/{DEMO_CONFIG.WWW_ENDPOINT}/<file_path:path>')
def server_static(file_path="index.html"):
	last_part = file_path[file_path.rfind('/') + 1:]
	all_files = _get_list_of_files(".")

	# Check whether the ending is a name of a file, otherwise it is an input text
	if (" " in last_part) or (file_path not in all_files):
		file_path = file_path.replace(last_part, "index.html")

	return static_file(file_path, root='./')


@route(f'/{DEMO_CONFIG.WWW_ENDPOINT}/feedback/', method='POST')
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


@route(f'/{DEMO_CONFIG.WWW_ENDPOINT}/get_random_example/', method='GET')
def get_random_example():
	return {"text": _get_random_sentence(example_sentences)}


def _parse_text(text: str, consider_tags: bool):
	if consider_tags:
		clean_text, tagged_ranges = parse_specified_tags(text)
		ud_parsed_text = ud_parser(clean_text)
		return ud_parsed_text, tagged_ranges

	return ud_parser(text), {}


def _extract_inner(
		text: str, args_extractor: ArgsExtractor, consider_tags: bool, document_id: str = None,
		limited_postags: Optional[List[POSTag]] = None, predicate_index: Optional[int] = None,
		extractions_info: Optional[DynamicExtractionsInfo] = None,
		rename_to_verbal_active=False
) -> Tuple[Extractions, Extractions, DynamicExtractionsInfo]:
	extractions_info = DynamicExtractionsInfo() if extractions_info is None else extractions_info
	document_id = str(uuid4()) if document_id is None else document_id

	ud_parsed_text, tagged_ranges = _parse_text(text, consider_tags)
	tagged_ranges = translate_char_ranges_to_word_ranges(ud_parsed_text, tagged_ranges) if consider_tags else None
	limited_indices = [predicate_index] if predicate_index is not None else None

	all_extractions = []
	tagged_extractions = []

	for parsed_sent in ud_parsed_text.sents:
		parsed_sent = parsed_sent.as_standalone_parsed_text()
		multi_word_extraction = args_extractor.extract_multiword(
			parsed_sent, limited_postags=limited_postags, limited_indices=limited_indices)
		# reduced_exts_by_idx = {idx: reduce_extractions_by_arg_types(multi_word_extraction.extractions, ) for idx in multi_word_extraction.extractions_per_idx}
		# multi_word_extraction = MultiWordExtraction(words=multi_word_extraction.words, extractions_per_idx=reduced_exts_by_idx)
		all_extractions += multi_word_extraction.extractions

		_tagged_extractions, extractions_info = generate_args_extraction_info(
			parsed_sent, multi_word_extraction, document_id,
			tagged_ranges=tagged_ranges, extractions_info=extractions_info,
			rename_to_verbal_active=rename_to_verbal_active)
		tagged_extractions += _tagged_extractions

	return all_extractions, tagged_extractions, extractions_info


@route(f'/{DEMO_CONFIG.WWW_ENDPOINT}/extract/', method='POST')
@force_min_response_time
def extract():
	text = request.json["text"]
	extraction_mode = request.json["extraction-mode"]
	limited_postags = request.json["limited-postags"]
	consider_tags = request.json["consider-tags"]
	args_extractor = args_extractors[extraction_mode]
	_, _, extractions_info = _extract_inner(
		text, args_extractor, consider_tags=consider_tags, limited_postags=limited_postags,
		rename_to_verbal_active=True)
	return asdict(extractions_info)


@route(f'/{DEMO_CONFIG.WWW_ENDPOINT}/match_references/', method='POST')
@force_min_response_time
def match_references():
	text = request.json["text"]
	extraction_mode = request.json["extraction-mode"]
	predicate_index = request.json["predicate-index"]
	consider_tags = request.json["consider-tags"]

	if extraction_mode == ExtractorType.NOMLEX:
		return {}

	args_extractor = args_extractors[extraction_mode]
	extractions, _, _ = _extract_inner(
		text, args_extractor, consider_tags=consider_tags, predicate_index=predicate_index,
		rename_to_verbal_active=False)

	best_scored_references = []
	for ext in extractions:
		verb = verb_translator.translate(ext.predicate_lemma, ext.predicate_postag)
		for arg in ext.args:
			scored_references = get_closest_references(
				arg, references_by_verb[verb], k_closest=1,
				similarity_scorer=EXTRACTORS_CONFIG.DEFAULT_METHOD_PARAMS.similarity_scorer,
				arg_types=[arg.arg_type])

			for scored_reference in scored_references:
				new_ext = reconstruct_extraction(scored_reference.extraction)
				new_ext.args = [ref_arg for ref_arg in new_ext.args if arg.arg_type == ref_arg.arg_type]
				best_scored_references.append(ScoredReference(extraction=new_ext, score=scored_reference.score))

	references_extractions_info = DynamicExtractionsInfo()
	document_id = str(uuid4())

	best_scored_references = sorted(best_scored_references, key=lambda _scored_ref: _scored_ref.score, reverse=True)
	for scored_ref in best_scored_references[:DEMO_CONFIG.MAX_MATCHING_RESULTS]:
		words = scored_ref.extraction.words
		ud_parsed_text = words if isinstance(words, ParsedText) else _parse_text(" ".join(words), consider_tags)[0]
		scored_ref.extraction.words = ud_parsed_text
		multi_word_extraction = MultiWordExtraction(
			words=ud_parsed_text,
			extractions_per_idx={scored_ref.extraction.predicate_idx: [scored_ref.extraction]})

		_, references_extractions_info = generate_args_extraction_info(
			ud_parsed_text, multi_word_extraction, document_id,
			extractions_info=references_extractions_info, rename_to_verbal_active=True)

	return asdict(references_extractions_info)


# @route(f'/{DEMO_CONFIG.WWW_ENDPOINT}/match/', method='POST')
# @force_min_response_time
# def match():
# 	text = request.json["text"]
# 	extraction_based = request.json["extraction-based"]
# 	args_extractor = args_extractors[extraction_based]
# 	tagged_extractions, _ = _extract_inner(text, args_extractor)
#
# 	dynamic_extractions_info = DynamicExtractionsInfo()
#
# 	document_id = str(uuid4())
# 	shuffle(parsed_example_documents)
# 	for parsed_text in parsed_example_documents:
# 		_, dynamic_extractions_info = generate_args_extraction_info(
# 			parsed_text, args_extractor, document_id,
# 			references=tagged_extractions, reference_matcher=extraction_matcher,
# 			dynamic_extractions_info=dynamic_extractions_info
# 		)
#
# 		if len(dynamic_extractions_info.mentions_by_event) > DEMO_CONFIG.MAX_MATCHING_RESULTS:
# 			break
#
# 	return asdict(dynamic_extractions_info)


if __name__ == '__main__':
	# Email for feedback response
	email_address = "" #input("Enter your e-mail address: ")
	password = "" #getpass("Password for sending emails: ")

	ud_parser = DependencyParserFactory(parser_name=PARSING_CONFIG.PARSER_NAME)()

	nomlex_version = NomlexVersion.V2
	verb_translator = VerbTranslatorFactory(nomlex_version=nomlex_version)()
	arg_encoder = ArgumentEncoderFactory(
		encoding_framework=ENCODING_CONFIG.ENCODING_FRAMEWORK,
		encoder_name=ENCODING_CONFIG.ENCODER_NAME,
		encoding_level=EncodingLevel.HEAD_IDX_IN_ARG_CONTEXT)()

	# Assuming that both methods are pp-typed and require normalized vectors
	references_by_verb = get_references_by_predicate(
		extractions=load_extraction_references(
			path=EXTRACTORS_CONFIG.REFERENCES_PATH_BY_ENCODER[arg_encoder_to_tuple_id(arg_encoder)],
			extracted_file_handler=ExtractedFileHandler(ud_parser),
			consider_pp_type=EXTRACTORS_CONFIG.DEFAULT_METHOD_PARAMS.consider_pp_type
		),
		verb_translator=verb_translator,
		normalize=EXTRACTORS_CONFIG.DEFAULT_METHOD_PARAMS.already_normalized
	)

	args_extractors = _get_arg_extractors()
	example_sentences = _get_examples()

	run(host=DEMO_CONFIG.URL, reloader=False, port=DEMO_CONFIG.PORT, server='paste')
