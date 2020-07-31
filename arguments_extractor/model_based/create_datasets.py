import zlib
from os.path import isfile
from collections import defaultdict

from tqdm import tqdm
from spacy.tokens import DocBin
import pandas as pd

from arguments_extractor.arguments_extractor import ArgumentsExtractor
from arguments_extractor.constants.lexicon_constants import *
from arguments_extractor.utils import flatten, difference_list, get_dependency_tree, ud_parser
from arguments_extractor import config

# Arguments dataset properties
ARG_DATASET_SIZE = 100000 # the number of wanted argument samples
ARG_MIN_N_ARGUMENTS = 1 # The minimum number of arguments for each extraction

# Example dataset properties
EXAMPLE_DATASET_SIZE = 100000 # the number of wanted example sentences
EXAMPLE_MIN_N_ARGUMENTS = 2 # The minimum number of arguments for each extraction

# Other properties (always relevant)
SENTENCE_LEN_LIMIT = 30

STATUS_PERIOD = 1000


# Counting the number of arguments for each complement type
count_for_argument = {COMP_SUBJ: 0, COMP_OBJ: 0, COMP_IND_OBJ: 0, COMP_PP: 0, COMP_NONE: 0}

def clean_sentence(sentence):
	# Avoiding underscore, cause it appears a lot on the wikipedia dataset
	sentence = sentence.replace("_", " ").strip(" \t\n\r")

	# Replace multi-whitespaces to a single one
	sentence = ' '.join(sentence.split())

	return sentence

def argument_to_sample(referenced_token, argument_span, suitable_verb, complement_type):
	doc_to_text = lambda doc: " ".join([token.orth_ for token in doc])
	return doc_to_text(referenced_token.doc), argument_span[0].i, argument_span[-1].i, referenced_token.i, suitable_verb, complement_type

def add_arguments(arguments_extractor, predicates_extractions, arguments_per_verb, verbs_per_complement_types, is_verb=False):
	n_added_arguments = 0

	for referenced_token, extractions in predicates_extractions.items():
		if is_verb:
			suitable_verb = arguments_extractor.verb_lexicon.find(referenced_token)[1]
		else:
			suitable_verb = arguments_extractor.nom_lexicon.find(referenced_token)[1]

		important_arguments = defaultdict(list)
		arguments_per_complement = defaultdict(list)
		for extraction in extractions:
			for complement_type in difference_list(extraction.keys(), [COMP_NONE]):
				if complement_type in verbs_per_complement_types.keys():
					important_arguments[extraction[complement_type]].append(complement_type)
					arguments_per_complement[complement_type].append(extraction[complement_type])

			for argument in extraction[COMP_NONE]:
				important_arguments[argument].append(COMP_NONE)

		chosen_arguments = []

		# Every chosen argument should have only one extraction option
		for argument, complement_types in important_arguments.items():
			chosen_arguments.append(argument)

			if len(complement_types) == 2 and COMP_NONE in complement_types:
				complement_types.remove(COMP_NONE)

			# Avoiding arguments that might have multiple possible complement types
			# And arguments that aren't the only candidate for a certain complement type (like two objects)
			complement_types = list(set(complement_types))
			if len(complement_types) == 1 and (complement_types[0] == COMP_NONE or len(arguments_per_complement[complement_types[0]]) == 1):
				complement_type = complement_types[0]
				if complement_type in [COMP_PP1, COMP_PP2]:
					complement_type = COMP_PP

				# Every argument\label should have the same precentage of samples in the datasets
				if count_for_argument[complement_type] <= ARG_DATASET_SIZE / len(count_for_argument.keys()):
					count_for_argument[complement_type] += 1
					arguments_per_verb[suitable_verb] += [argument_to_sample(referenced_token, argument, suitable_verb, complement_type)]
					n_added_arguments += 1
				elif complement_type in verbs_per_complement_types.keys():
					verbs_per_complement_types.pop(complement_type)
					if complement_type == COMP_PP:
						verbs_per_complement_types.pop(COMP_PP1)
						verbs_per_complement_types.pop(COMP_PP2)

					verbs_per_complement_types["TOTAL"] = list(set(flatten(verbs_per_complement_types.values())))

	return n_added_arguments

def aggregate_arguments(dataset_path, arguments_extractor: ArgumentsExtractor, verbs_per_complement_types: dict):
	arguments_per_verb = defaultdict(list)
	total_founded_arguments = 0
	total_verb_arguments = 0
	total_nom_arguments = 0
	total_number_of_sentences = 0

	# Load the parsed dataset
	# Ignore files with existing arguments files (when not forced)
	parsed_dataset = None
	try:
		if not config.IGNORE_PROCESSED_DATASET or not isfile(dataset_path.replace(".parsed", "_args.csv")):
			with open(dataset_path, "rb") as parsed_dataset_file:
				dataset_bytes = parsed_dataset_file.read()
				doc_bin = DocBin().from_bytes(dataset_bytes)
				parsed_dataset = doc_bin.get_docs(ud_parser.vocab)
	except zlib.error:
		pass

	if parsed_dataset is None:
		return None

	try:
		for doc in parsed_dataset:
			# Get the extractions
			verb_extractions, nom_extractions = arguments_extractor.extract_arguments(doc, min_arguments=ARG_MIN_N_ARGUMENTS, specify_none=True, trim_arguments=False, limited_verbs=verbs_per_complement_types["TOTAL"])

			# Aggregate the relevant arguments
			n_verb_arguments = add_arguments(arguments_extractor, verb_extractions, arguments_per_verb, verbs_per_complement_types, is_verb=True)
			n_nom_arguments = add_arguments(arguments_extractor, nom_extractions, arguments_per_verb, verbs_per_complement_types, is_verb=False)
			total_verb_arguments += n_verb_arguments
			total_nom_arguments += n_nom_arguments
			total_founded_arguments += n_nom_arguments + n_verb_arguments

			# Add to chosen sentences only if a relevant argument argument was found within
			if n_verb_arguments + n_nom_arguments > 0:
				# Print the progress every X number of new argument samples
				if float(int(total_founded_arguments / STATUS_PERIOD)) == total_founded_arguments / STATUS_PERIOD:
					print(dataset_path.split('/')[-1] + f" (Total: {total_founded_arguments}- {list(count_for_argument.items())})")

				total_number_of_sentences += 1

			if total_founded_arguments >= ARG_DATASET_SIZE:
				break

	except KeyboardInterrupt:
		print("The dataset creation was paused, but saved!")

	print(f"The number of needed sentences: {total_number_of_sentences}")
	print(f"The dataset contains {total_founded_arguments} arguments, including {total_verb_arguments} arguments of VERB and {total_nom_arguments} arguments of NOM")

	return arguments_per_verb

def create_args_dataset(dataset_path, arguments_extractor: ArgumentsExtractor):
	limited_complement_types = [COMP_SUBJ, COMP_OBJ, COMP_IND_OBJ, COMP_PP, COMP_PP1, COMP_PP2, COMP_NONE]
	verbs_per_complement_types = defaultdict(list)

	# Find the limited verbs for each complement type
	for complement_type in limited_complement_types:
		verbs_per_complement_types[complement_type] = arguments_extractor.nom_lexicon.find_verbs(complement_type)

	verbs_per_complement_types["TOTAL"] = list(set(flatten(verbs_per_complement_types.values())))

	arguments_per_verb = aggregate_arguments(dataset_path, arguments_extractor, verbs_per_complement_types)

	if arguments_per_verb is not None:
		arguments_dataset = pd.DataFrame(flatten(arguments_per_verb.values()))
		arguments_dataset.to_csv(dataset_path.replace(".parsed", "_args.csv"), sep="\t", index=False, header=None)

def create_parsed_dataset(sentences_path):
	parsed_dataset = DocBin(attrs=["ID", "ORTH", "LEMMA", "TAG", "POS", "HEAD", "DEP", "ENT_IOB", "ENT_TYPE"], store_user_data=True)

	# Load the sentences dataset (".txt")
	# Ignore files with existing parsed files (when not forced)
	sentences_dataset = None
	if not config.IGNORE_PROCESSED_DATASET or not isfile(sentences_path.replace(".txt", ".parsed")):
		sentences_dataset = open(sentences_path, "r")

	if sentences_dataset is None:
		return

	count = 0
	try:
		for sentence in sentences_dataset:
			count += 1

			# Print the progress every X number of parsed sentences
			if float(int(count / STATUS_PERIOD)) == count / STATUS_PERIOD:
				print(sentences_path.split('/')[-1] + f" ({count})")

			if type(sentence) == str:
				sentence = clean_sentence(sentence)

				if sentence == "" or len(sentence.split(" ")) >= SENTENCE_LEN_LIMIT:
					continue

				doc = get_dependency_tree(sentence)
			else: # the sentence is actually the dependency tree
				doc = sentence

			# The current line must include only one sentence
			sentences = list(doc.sents)
			if len(sentences) > 1:
				continue

			parsed_dataset.add(doc)

	except KeyboardInterrupt:
		print("The parsing was paused, but saved!")

	with open(sentences_path.replace(".txt", ".parsed"), "wb") as parsed_file:
		parsed_file.write(parsed_dataset.to_bytes())



def create_examples_dataset(input_dataset_path, output_dataset_path, arguments_extractor: ArgumentsExtractor):
	# Load the shuffled wikipedia dataset (".txt")
	# Ignore files with existing parsed files (when not forced)
	sentences_dataset = open(input_dataset_path, "r")

	# Choose some "interesting sentences" (= sentences with at least one exrtraction of two or more arguments)
	chosen_dataset = []
	status_bar = tqdm(sentences_dataset, "Choosing example sentences", leave=True)
	for sentence in status_bar:
		sentence = clean_sentence(sentence)

		if sentence == "" or len(sentence.split(" ")) >= SENTENCE_LEN_LIMIT:
			continue

		doc = get_dependency_tree(sentence)

		# The current line must include only one sentence
		sentences = list(doc.sents)
		if len(sentences) > 1:
			continue

		sentence = sentences[0]

		# Get the extractions
		verb_extractions, nom_extractions = arguments_extractor.rule_based_extraction(doc, min_arguments=EXAMPLE_MIN_N_ARGUMENTS)

		if verb_extractions != {} or nom_extractions != {}:
			chosen_dataset.append(sentence.orth_ + "\n")

		status_bar.set_description(f"Choosing example sentences ({len(chosen_dataset)})")

		if len(chosen_dataset) > EXAMPLE_DATASET_SIZE:
			break

	# Save the chosen sentences
	with open(output_dataset_path, "w") as target_file:
		target_file.writelines(chosen_dataset)