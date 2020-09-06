import random
import zlib
from os.path import isfile
from collections import defaultdict

import pandas as pd
from spacy.tokens import DocBin, Token

from arguments_extractor.arguments_extractor import ArgumentsExtractor
from arguments_extractor.constants.lexicon_constants import *
from arguments_extractor.constants.ud_constants import *
from arguments_extractor.constants.dataset_constants import ARGS_LABELS, NOUNS_LABELS
from arguments_extractor.utils import flatten, get_dependency_tree, ud_parser
from arguments_extractor import config

class DatasetCreator:
	STATUS_PERIOD = 1000
	SENT_LEN_LIMIT = 30 # Relevant in parsing

	args_extractor: ArgumentsExtractor
	min_n_arguments: int
	dataset_size: int

	def __init__(self, args_extractor:ArgumentsExtractor, dataset_size=100000, min_n_arguments=1):
		self.args_extractor = args_extractor
		self.verb_noun_matcher = args_extractor.verb_noun_matcher
		self.dataset_size = dataset_size
		self.min_n_arguments = min_n_arguments

	@staticmethod
	def clean_sentence(sent):
		# Avoiding underscore, cause it appears a lot on the wikipedia dataset
		sent = sent.replace("_", " ").strip(" \t\n\r")

		# Replace multi-whitespaces to a single one
		sent = ' '.join(sent.split())

		return sent

	def parse_sentence(self, sent):
		if type(sent) == str:
			sent = self.clean_sentence(sent)

			if sent == "" or len(sent.split(" ")) >= self.SENT_LEN_LIMIT:
				return None

			doc = get_dependency_tree(sent)

		else:  # the sentence is actually the dependency tree
			doc = sent

		# The current line must include only one sentence
		if len(list(doc.sents)) > 1:
			return None

		return doc

	@staticmethod
	def load_dataset(input_path, output_path, binary=False):
		# Load the dataset in the given path, and ignore it if the given output exists
		dataset = None

		if config.IGNORE_PROCESSED_DATASET and isfile(output_path):
			return None

		try:
			if not binary:
				return open(input_path, "r")

			with open(input_path, "rb") as parsed_dataset_file:
				dataset_bytes = parsed_dataset_file.read()
				doc_bin = DocBin().from_bytes(dataset_bytes)
				dataset = doc_bin.get_docs(ud_parser.vocab)
		except zlib.error:
			pass

		return dataset

	@staticmethod
	def doc_to_text(doc):
		return " ".join([token.orth_ for token in doc])

	def check_label(self, labels_counts, label):
		# Returns whether we should add more example of the given label
		# The function also updates the label's count

		if label not in labels_counts:
			return False

		if labels_counts[label] > self.dataset_size / len(labels_counts):
			return False

		labels_counts[label] += 1

		return True

	def sample_to_tuple(self, doc, predicate:Token, verb, label, arg=None):
		sent = self.doc_to_text(doc)
		is_verb = predicate.pos_ == UPOS_VERB

		arg_head_idx = arg.root.i if arg else -1
		arg_start_idx = arg[0].i if arg else -1
		arg_end_idx = arg[-1].i if arg else -1
		arg_str = self.doc_to_text(doc[arg_start_idx:arg_end_idx]) if arg else ""

		verb_types, nom_types = [], [] #self.verb_noun_matcher.get_possible_types(verb)
		verb_types = " ".join(verb_types)
		nom_types = " ".join(nom_types)

		return sent, predicate.i, predicate.lemma, \
			   arg_head_idx, arg_start_idx, arg_end_idx, arg_str, \
			   is_verb, verb, verb_types, nom_types, \
			   label

	def print_status(self, samples, dataset_path, labels_count=None):
		n_samples = len(samples)

		if float(int(n_samples / self.STATUS_PERIOD)) == n_samples / self.STATUS_PERIOD:
			if labels_count is None:
				print(dataset_path.split('/')[-1] + f" ({n_samples})")
			else:
				print(dataset_path.split('/')[-1] + f" (Total: {n_samples}- {list(labels_count.items())})")



	def get_args_samples(self, doc, labels_counts, labels_verbs):
		# Get the extractions
		extractions = self.args_extractor.extract_arguments(doc, min_arguments=self.min_n_arguments,
															specify_none=True, trim_arguments=False,
															limited_verbs=labels_verbs["TOTAL"], return_single=True)
		args_samples = []

		# Aggregate the relevant arguments
		for predicate, extractions in extractions.items():
			verb = self.verb_noun_matcher.get_suitable_verb(predicate)

			types_per_arg = defaultdict(set)
			args_per_type = defaultdict(set)
			for extraction in extractions:

				for arg_type, arg in extraction.items():
					if arg_type in labels_counts.keys():
						types_per_arg[arg].update(arg_type)
						args_per_type[arg_type].update(arg)

			for arg, types in types_per_arg.items():
				if len(types) == 2 and COMP_NONE in types:
					types.remove(COMP_NONE)

				# The argument should have only one possible type
				if len(types) > 1:
					continue

				# The type should be appropriate for only one argument
				arg_type = list(types)[0]
				possible_args = args_per_type[arg_type]
				if arg_type != COMP_NONE and len(possible_args) != 1:
					continue

				if arg_type in [COMP_PP1, COMP_PP2]:
					arg_type = COMP_PP

				if self.check_label(labels_counts, arg_type):
					sample = self.sample_to_tuple(doc, predicate, arg, verb, arg_type)
					args_samples.append(sample)

				elif arg_type in labels_verbs.keys():
					labels_verbs.pop(arg_type)
					if arg_type == COMP_PP:
						labels_verbs.pop(COMP_PP1)
						labels_verbs.pop(COMP_PP2)

					labels_verbs["TOTAL"] = list(set(flatten(labels_verbs.values())))

		return args_samples

	def get_nouns_samples(self, doc, labels_counts, labels_verbs=None):
		common_nouns = []
		noun_samples = []

		for word in doc:
			# Ignore proper nouns and words from other part of speech
			if word.pos_ != UPOS_NOUN or word.tag_ in [TAG_NNPS, TAG_NNS]:
				continue

			# Try to find the noun in the lexicon
			nom_entry, verb = self.args_extractor.nom_lexicon.find(word, verb_noun_matcher=self.verb_noun_matcher, be_certain=True)

			# We found a noun that isn't even an estimated nom
			if verb is None:
				common_nouns.append(word)
				continue

			# We can only use noms with one type
			nom_types = nom_entry.get_nom_types(ignore_part=True)
			if len(nom_types) != 1:
				continue

			nom_type = nom_types[0]

			if self.check_label(labels_counts, nom_type):
				sample = self.sample_to_tuple(doc, word, None, verb, nom_type)
				noun_samples.append(sample)

		# Choose one random noun from the sentence
		if common_nouns != [] and self.check_label(labels_counts, COMP_NONE):
			noun = random.choice(common_nouns)
			sample = self.sample_to_tuple(doc, noun, None, COMP_NONE, COMP_NONE)
			noun_samples.append(sample)

		return noun_samples

	def create_dataset(self, in_dataset_path, out_dataset_path, get_samples_func, labels, labels_verbs=None):
		# Create a dataset according to the given properites
		# The input dataset should already be parsed
		in_dataset = self.load_dataset(in_dataset_path, out_dataset_path, binary=True)

		if in_dataset is None:
			return None

		samples = []
		labels_counts = {l:0 for l in labels}

		for doc in in_dataset:
			new_samples = get_samples_func(doc, labels_counts, labels_verbs)
			samples += new_samples

			if new_samples == []:
				continue

			self.print_status(samples, in_dataset_path, labels_counts)
			if len(samples) >= self.dataset_size:
				break

		print(f"The dataset contains {len(samples)} samples")

		out_dataset = pd.DataFrame(samples)
		out_dataset.to_csv(out_dataset_path, sep="\t", index=False, header=False)

	def parse_dataset(self, in_dataset_path, out_dataset_path, save_as_str=False, condition_func=None):
		in_dataset = self.load_dataset(in_dataset_path, out_dataset_path, binary=False)

		if in_dataset is None:
			return None

		sents = []
		doc_bin = DocBin(attrs=["ID", "ORTH", "LEMMA", "TAG", "POS", "HEAD", "DEP", "ENT_IOB", "ENT_TYPE"],
								store_user_data=True)

		for sent in in_dataset:
			doc = self.parse_sentence(sent)

			if doc is None:
				continue

			if condition_func and condition_func(doc):
				continue

			doc_bin.add(doc)
			sents.append(sent)

			self.print_status(in_dataset_path, doc_bin)
			if len(doc_bin) >= self.dataset_size:
				break

		print(f"The dataset contains {len(doc_bin)} parsed sentences")

		# Save the resulted dataset as strings or as parsed
		if save_as_str:
			with open(out_dataset_path, "w") as target_file:
				target_file.writelines(sents)
		else:
			with open(out_dataset_path, "wb") as parsed_file:
				parsed_file.write(doc_bin.to_bytes())



	def create_args_dataset(self, in_dataset_path):
		# Create the ARGS dataset
		out_dataset_path = in_dataset_path.replace(".parsed", "_args.csv")

		# Find the limited verbs for each complement type
		labels_verbs = defaultdict(list)
		for arg_type in ARGS_LABELS + [COMP_PP1, COMP_PP2]:
			labels_verbs[arg_type] = self.args_extractor.nom_lexicon.get_verbs_by_arg(arg_type)

		# The total verbs that can appear in the dataset
		# This entry will be updated whenever we sample enough examples of certain labels
		labels_verbs["TOTAL"] = list(set(flatten(labels_verbs.values())))

		self.create_dataset(in_dataset_path, out_dataset_path, self.get_args_samples, ARGS_LABELS, labels_verbs)

	def create_nouns_dataset(self, in_dataset_path):
		# Create the NOUNS dataset
		out_dataset_path = in_dataset_path.replace(".parsed", "_nouns.csv")
		self.create_dataset(in_dataset_path, out_dataset_path, self.get_nouns_samples, NOUNS_LABELS, None)

	def create_parsed_dataset(self, in_dataset_path):
		# Create a parsed dataset from wikipedia
		out_dataset_path = in_dataset_path.replace(".txt", ".parsed")
		self.parse_dataset(in_dataset_path, out_dataset_path, save_as_str=False)

	def create_examples_dataset(self, in_dataset_path, out_dataset_path):
		# Create a dataset of good example sentences

		# Choose only sentences with at least one NOMLEX predicate
		self.min_n_arguments = 2
		condition_func = lambda doc: self.args_extractor.extract_arguments(doc, min_arguments=self.min_n_arguments, return_single=True) != {}
		self.parse_dataset(in_dataset_path, out_dataset_path, condition_func=condition_func, save_as_str=True)