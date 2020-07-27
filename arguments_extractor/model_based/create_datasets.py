import random
from os.path import isfile
from io import TextIOWrapper

from tqdm import tqdm
from spacy.tokens import DocBin
import pandas as pd

from arguments_extractor.arguments_extractor import *
from arguments_extractor.utils import *

# Arguments dataset properties
ARG_DATASET_SIZE = 40000 # the number of arguments
ARG_MIN_N_ARGUMENTS = 1 # The minimum number of arguments for each extraction
ARG_LEXICON_RATIO = 0.8

# Example dataset properties
EXAMPLE_DATASET_SIZE = 100000 # the number of sentences
EXAMPLE_MIN_N_ARGUMENTS = 2 # The minimum number of arguments for each extraction

# Other properties (always relevant)
SENTENCE_LEN_LIMIT = 30


count_for_argument = {COMP_SUBJ: 0, COMP_OBJ: 0, COMP_IND_OBJ: 0, COMP_PP: 0}

def load_dataset(base_dataset_path, chosen_dataset_path):
	dataset = None

	if config.LOAD_DATASET:
		parsed_dataset_path = chosen_dataset_path.replace(".txt", ".parsed")
		if isfile(parsed_dataset_path):
			with open(parsed_dataset_path, "rb") as parsed_dataset_file:
				dataset_bytes = parsed_dataset_file.read()
				doc_bin = DocBin().from_bytes(dataset_bytes)
				dataset = doc_bin.get_docs(ud_parser.vocab)

		elif isfile(chosen_dataset_path):
			dataset = open(chosen_dataset_path, "r")

	if dataset is None:
		dataset = open(base_dataset_path, "r")

	return dataset

def split_arguments(arguments_per_verb, ratio):
	verbs = list(arguments_per_verb.keys())
	print(verbs)
	print(len(verbs))
	random.shuffle(verbs)

	train_part = int(len(verbs) * ratio)
	train_limited_verbs = verbs[:train_part]
	test_limited_verbs = verbs[train_part:]

	doc_to_text = lambda doc: " ".join([token.orth_ for token in doc])

	train_arguments = flatten([arguments_per_verb[verb] for verb in train_limited_verbs])
	train_set = pd.DataFrame([(doc_to_text(token.doc), argument[0].i, argument[-1].i, token.i, suitable_verb, complement_type) for suitable_verb, token, complement_type, argument in train_arguments])

	test_arguments = flatten([arguments_per_verb[verb] for verb in test_limited_verbs])
	test_set = pd.DataFrame([(doc_to_text(token.doc), argument[0].i, argument[-1].i, token.i, suitable_verb, complement_type) for suitable_verb, token, complement_type, argument in test_arguments])

	return train_set, test_set

def add_arguments(arguments_extractor, verb_extractions, extractions_per_verb, limited_complement_types, is_verb=False):
	n_added_arguments = 0

	for word_token, extractions in verb_extractions.items():
		if is_verb:
			suitable_verb = arguments_extractor.verb_lexicon.find(word_token).orth
		else:
			suitable_verb = arguments_extractor.nom_lexicon.find(word_token).verb.split("#")[0]

		important_arguments = defaultdict(list)
		for extraction in extractions:
			for complement_type in extraction.keys():
				if complement_type in limited_complement_types:
					important_arguments[extraction[complement_type]].append(complement_type)

		# Every chosen argument should have only one extraction option
		for argument, complement_types in important_arguments.items():
			# Avoiding arguments that might have multiple possible complement types
			complement_types = list(set(complement_types))
			if len(complement_types) == 1:
				complement_type = complement_types[0]
				if complement_type in [COMP_PP1, COMP_PP2]:
					complement_type = COMP_PP

				if count_for_argument[complement_type] <= ARG_DATASET_SIZE / len(count_for_argument.keys()):
					count_for_argument[complement_type] += 1

					extractions_per_verb[suitable_verb] += [(suitable_verb, word_token, complement_type, argument)]
					n_added_arguments += 1

	return n_added_arguments

def aggregate_arguments(arguments_extractor: ArgumentsExtractor, limited_complement_types: list):
	dataset = load_dataset(config.WIKI_SENTENCES_PATH, config.ARG_SENTENCES_PATH)

	chosen_dataset = []
	parsed_chosen_dataset = DocBin(attrs=["ID", "ORTH", "LEMMA", "TAG", "POS", "HEAD", "DEP", "ENT_IOB", "ENT_TYPE"], store_user_data=True)
	arguments_per_verb = defaultdict(list)
	total_founded_arguments = 0
	total_verb_arguments = 0
	total_nom_arguments = 0

	status_bar = tqdm(dataset, "Creating arguments datasets", leave=True)
	for sentence in status_bar:
		if type(sentence) == str:
			sentence = sentence.strip(" \t\n\r")

			if sentence == "" or len(sentence.split(" ")) >= SENTENCE_LEN_LIMIT:
				continue

			doc = get_dependency_tree(sentence)
		else:  # sentence is the dependency tree
			doc = sentence

		# The current line must include only one sentence
		sentences = list(doc.sents)
		if len(sentences) > 1:
			continue

		sentence = sentences[0]

		# Get the extractions
		verb_extractions, nom_extractions = arguments_extractor.rule_based_extraction(doc, min_arguments=ARG_MIN_N_ARGUMENTS)

		# Aggregate the relevant arguments
		n_verb_arguments = add_arguments(arguments_extractor, verb_extractions, arguments_per_verb, limited_complement_types, is_verb=True)
		n_nom_arguments = add_arguments(arguments_extractor, nom_extractions, arguments_per_verb, limited_complement_types, is_verb=False)
		total_verb_arguments += n_verb_arguments
		total_nom_arguments += n_nom_arguments
		total_founded_arguments += n_nom_arguments + n_verb_arguments
		status_bar.set_description(f"Creating arguments datasets (Total: {total_founded_arguments}- {count_for_argument.items()})")

		parsed_chosen_dataset.add(doc)
		chosen_dataset.append(sentence.orth_ + "\n")

		if total_founded_arguments >= ARG_DATASET_SIZE:
			break

	print(f"NOM: {total_verb_arguments}")
	print(f"VERB: {total_nom_arguments}")

	if type(dataset) == TextIOWrapper:
		dataset.close()

	if not config.LOAD_DATASET:
		with open(config.ARG_SENTENCES_PATH, "w") as target_file:
			target_file.writelines(chosen_dataset)

	if not config.LOAD_DATASET:
		with open(config.ARG_SENTENCES_PATH.replace('.txt', '.parsed'), "wb") as parsed_file:
			parsed_file.write(parsed_chosen_dataset.to_bytes())

	print(f"The number of sentences in the chosen dataset: {len(chosen_dataset)}")
	print(f"The number of extractions in the chosen dataset: {total_founded_arguments}")

	return arguments_per_verb

def create_args_datasets(arguments_extractor: ArgumentsExtractor):
	config.CLEAN_NP = False
	arguments_per_verb = aggregate_arguments(arguments_extractor, [COMP_SUBJ, COMP_OBJ, COMP_IND_OBJ, COMP_PP, COMP_PP1, COMP_PP2])
	# create_statistics(extractions_per_verb)

	train_set, test_set = split_arguments(arguments_per_verb, ARG_LEXICON_RATIO)

	print(f"The number of extractions in the train set: {len(train_set)}")
	print(f"The number of extractions in the test set: {len(test_set)}")

	train_set.to_csv(config.ARG_DATASET_DIR + "/train", sep="\t")
	test_set.to_csv(config.ARG_DATASET_DIR + "/test", sep= "\t")



def create_examples_dataset(arguments_extractor: ArgumentsExtractor):
	# Loading the shuffled wikipedia dataset
	dataset = load_dataset(config.WIKI_SENTENCES_PATH, "")

	# Choose some "interesting sentences" (= sentences with at least one exrtraction of two or more arguments)
	chosen_dataset = []
	status_bar = tqdm(dataset, "Choosing example sentences", leave=True)
	for sentence in status_bar:
		sentence = sentence.strip(" \t\n\r")

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
	with open(config.EXAMPLE_SENTENCES_PATH, "w") as target_file:
		target_file.writelines(chosen_dataset)