import os
import random
import datetime
import time
import zlib
from os.path import isfile, dirname, basename, join
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from spacy.tokens import DocBin, Token
from nltk.tokenize import sent_tokenize
from nlp import load_dataset
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42

from noun_as_verb.arguments_extractor import ArgumentsExtractor
from noun_as_verb.constants.lexicon_constants import *
from noun_as_verb.constants.ud_constants import *
from noun_as_verb.constants.dataset_constants import SYN_ARGS_LABELS, SYN_NOUNS_LABELS
from noun_as_verb.utils import flatten, get_dependency_tree, ud_parser, separate_line_print, is_proper_noun
from noun_as_verb import config

class DatasetCreator:
	LOG_INTERVAL = 10000 # Log status every such number of examples

	# Smaller or Larger sentences are filtered in parsing stage
	SENT_LEN_MAX = 35
	SENT_LEN_MIN = 3

	args_extractor: ArgumentsExtractor
	dataset_size: int

	def __init__(self, args_extractor:ArgumentsExtractor, dataset_size=-1):
		self.args_extractor = args_extractor
		self.verb_noun_matcher = args_extractor.verb_noun_matcher
		self.dataset_size = dataset_size # -1 means not limited

	@staticmethod
	def clean_sentence(sent):
		# Avoiding underscore, cause it appears a lot on the wikipedia dataset
		sent = sent.replace("_", " ").replace("\t", "").strip(" \t\r\n")

		# Replace multi-whitespaces to a single one
		sent = ' '.join(sent.split())

		return sent

	@staticmethod
	def is_english(text):
		# Determines whether the given text was written in English
		try:
			lang = detect(text)
			return lang == "en"
		except:
			return False

	@staticmethod
	def is_english_letters(s):
		try:
			s.encode(encoding='utf-8').decode('ascii')
		except UnicodeDecodeError:
			return False

		return True

	def parse_sentence(self, sent):
		if type(sent) == str:
			sent = self.clean_sentence(sent)
			n_words = len(sent.split(" "))

			# Ignore too short or too long sentences
			if sent == "" or n_words >= self.SENT_LEN_MAX or n_words <= self.SENT_LEN_MIN:
				return None

			if not self.is_english(sent):
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

		if self.dataset_size == -1:
			labels_counts[label] += 1
			return True

		#if labels_counts[label] > self.dataset_size / len(labels_counts):
		#	return False

		labels_counts[label] += 1

		return True

	def check_dataset_size(self, samples_so_far):
		if self.dataset_size == -1:
			return True

		return len(samples_so_far) <= self.dataset_size

	def sample_to_tuple(self, doc, predicate:Token, verb, label, arg=None):
		sent = self.doc_to_text(doc)
		is_verb = predicate.pos_ == UPOS_VERB

		arg_head_idx = arg.root.i if arg else -1
		arg_start_idx = arg[0].i if arg else -1
		arg_end_idx = arg[-1].i + 1 if arg else -1
		arg_str = self.doc_to_text(doc[arg_start_idx:arg_end_idx]) if arg else ""

		# verb_types, nom_types = self.verb_noun_matcher.get_possible_types(verb)
		# verb_types = " ".join(verb_types)
		# nom_types = " ".join(nom_types)

		sample_tuple = sent, predicate.i, predicate.lemma_.lower(), is_verb, verb, \
					   arg_head_idx, arg_start_idx, arg_end_idx, arg_str, label

		if config.DEBUG:
			print("------------------------------")
			print(doc)
			print(sample_tuple)

		return sample_tuple

	def print_status(self, n_sentences, dataset_path, start_time, labels_count=None):
		#if float(int(n_sentences / self.LOG_INTERVAL)) == n_sentences / self.LOG_INTERVAL:
		if n_sentences % self.LOG_INTERVAL != 0:
			return

		running_time = time.time() - start_time
		status = f"{basename(dataset_path)}, RunningTime={str(datetime.timedelta(seconds=running_time))}"

		if labels_count is None:
			status += f" ({n_sentences})"
		else:
			status += f" (Total Sentences: {n_sentences}- {list(labels_count.items())})"

		print(status)



	def articles_to_sents(self, wiki_articles):
		total_sents = []

		for article_info in tqdm(wiki_articles):
			text = article_info["text"]
			paras = text.split("\n_START_PARAGRAPH_\n")

			for para in paras:
				# Avoid article titles
				if "_START_ARTICLE_" in para:
					continue

				# Avoid section titles
				sents_str = para.split("_START_SECTION_")[0]
				sents_str = sents_str.replace("_NEWLINE_", "\n")
				sents = sents_str.split("\n")
				sents = flatten([sent_tokenize(sent) for sent in sents])
				sents = [self.clean_sentence(sent) + "\n" for sent in sents if sent != ""]
				total_sents += sents

			if not self.check_dataset_size(total_sents):
				break

		return total_sents

	def parse_dataset(self, in_dataset_path, out_dataset_path, save_as_str=False, condition_func=None):
		in_dataset = self.load_dataset(in_dataset_path, out_dataset_path, binary=False)

		if in_dataset is None:
			return None

		start_time = time.time()
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

			self.print_status(len(doc_bin), out_dataset_path, start_time)
			if not self.check_dataset_size(doc_bin):
				break

		print(f"The dataset {basename(out_dataset_path)} contains {len(doc_bin)} parsed sentences")

		os.makedirs(dirname(out_dataset_path), exist_ok=True)

		# Save the resulted dataset as string if needed
		if save_as_str:
			with open(out_dataset_path, "w") as target_file:
				target_file.writelines(sents)

		# Save the resulted parsed dataset
		with open(out_dataset_path, "wb") as parsed_file:
			parsed_file.write(doc_bin.to_bytes())

	def get_nouns_samples(self, doc, labels_counts, labels_related_verbs=None):
		common_nouns = []
		noun_samples = []

		for word in doc:
			# Ignore proper nouns and words from other part of speech
			if word.pos_ != UPOS_NOUN or is_proper_noun(word) or word.lemma_ == "-PRON-":
				continue

			# Ignore words that contain non-English letters
			if not self.is_english_letters(word.orth_):
				continue

			# Ignore words that contain signs (allow only alphabetic letters and one hyphen)
			if not word.orth_.replace("-", "").isalpha() or not word.lemma_.replace("-", "").isalpha():
				continue

			# Cannot start or end with hypen
			if word.orth_[0] == "-" or word.orth_[-1] == "-":
				continue

			# Try to find the noun in NOMLEX
			nom_entry, verb = self.args_extractor.nom_lexicon.find(word, be_certain=True)

			if verb is None:
				# Is this noun not even an estimated nom?
				estimated_verb = self.verb_noun_matcher.get_suitable_verb(word)
				if estimated_verb is None:
					common_nouns.append(word)

				continue

			# We can only use noms with one type
			nom_types = nom_entry.get_nom_types(ignore_part=True)
			if len(nom_types) != 1:
				continue

			nom_type = nom_types[0]

			if self.check_label(labels_counts, nom_type):
				sample = self.sample_to_tuple(doc, word, verb, nom_type)
				noun_samples.append(sample)

		# Choose one random noun from the sentence
		for noun in common_nouns:
			if self.check_label(labels_counts, "NOT-NOM"):
				#noun = random.choice(common_nouns)
				sample = self.sample_to_tuple(doc, noun, "NOT-NOM", "NOT-NOM")
				noun_samples.append(sample)

		return noun_samples

	def get_args_samples(self, doc, labels_counts, labels_related_verbs):
		# Get the extractions
		#extractions_per_word = self.args_extractor.extract_arguments(doc, min_arguments=1,
		#													specify_none=True, trim_arguments=False,
		#													return_single=True) #, limited_verbs=labels_related_verbs["TOTAL"])

		extractions_per_word = self.args_extractor.extract_arguments(doc, min_arguments=1,
																	 specify_none=True, trim_arguments=False,
																	 return_single=True, transer_args_predictor=self.args_extractor.transer_args_predictor) #, limited_verbs=labels_related_verbs["TOTAL"])

		# if config.DEBUG:
		# 	print("------------------------------")
		# 	print(doc)
		# 	separate_line_print(extractions_per_word)

		args_samples = []

		# Aggregate the relevant arguments
		for predicate, extractions in extractions_per_word.items():
			if predicate.pos_ == UPOS_VERB:
				_, verb = self.args_extractor.verb_lexicon.find(predicate, be_certain=True)
			else:
				_, verb = self.args_extractor.nom_lexicon.find(predicate, be_certain=True)

			# verb = self.verb_noun_matcher.get_suitable_verb(predicate)
			# Ignore nominal predicates that might be common nouns
			if verb is None:
				continue

			#assert verb, f"Couldn't find the verb for the predicate {predicate}, which appeared in NOMLEX!"

			types_per_arg, args_per_type = defaultdict(set), defaultdict(set)
			for extraction in extractions:
				for arg_type, arg in extraction.items():
					if arg_type in labels_counts.keys():
						args = [arg] if arg_type != COMP_NONE else arg
						[types_per_arg[arg].add(arg_type) for arg in args]
						args_per_type[arg_type].update(args)

			for arg, types in types_per_arg.items():
				# Arguments that correspond to the predicate do not appear in this dataset
				if arg.root.i == predicate.i:
					continue

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
					sample = self.sample_to_tuple(doc, predicate, verb, arg_type, arg)
					args_samples.append(sample)

				elif labels_related_verbs and arg_type in labels_related_verbs.keys():
					labels_related_verbs.pop(arg_type)
					if arg_type == COMP_PP:
						labels_related_verbs.pop(COMP_PP1)
						labels_related_verbs.pop(COMP_PP2)

					labels_related_verbs["TOTAL"] = list(set(flatten(labels_related_verbs.values())))

		args_samples += self.get_nouns_samples(doc, labels_counts, labels_related_verbs)

		return args_samples

	def create_custom_dataset(self, in_dataset_path, out_dataset_path, get_samples_func, labels, labels_related_verbs=None):
		# Create a dataset according to the given properites
		# The input dataset should already be parsed
		in_dataset = self.load_dataset(in_dataset_path, out_dataset_path, binary=True)

		if in_dataset is None:
			return None

		start_time = time.time()
		samples = []
		n_sentences = 0
		labels_counts = {l:0 for l in labels}

		for doc in in_dataset:
			#print("--------------------------------------------------------------------------------")
			#print(doc.text)
			new_samples = get_samples_func(doc, labels_counts, labels_related_verbs)
			samples += new_samples
			n_sentences += 1

			self.print_status(n_sentences, out_dataset_path, start_time, labels_counts)
			if not self.check_dataset_size(samples):
				break

		print(f"The dataset {basename(out_dataset_path)} contains {len(samples)} samples")

		# Save the resulted dataset as csv file
		os.makedirs(dirname(out_dataset_path), exist_ok=True)
		out_dataset = pd.DataFrame(samples)
		out_dataset.to_csv(out_dataset_path, sep="\t", index=False, header=False)



	def create_sentences_dataset(self, out_dataset_path):
		# Creates a dataset of sentences from wikipedia, using haggingface

		if config.IGNORE_PROCESSED_DATASET and isfile(out_dataset_path):
			return

		# Extract all the sentences from the wikipedia articles
		wiki_dataset_split = load_dataset("wiki40b", "en")
		train_sents = self.articles_to_sents(wiki_dataset_split["train"])
		val_sents = self.articles_to_sents(wiki_dataset_split["validation"])
		test_sents = self.articles_to_sents(wiki_dataset_split["test"])
		sents = train_sents + val_sents + test_sents
		print(f"The dataset {basename(out_dataset_path)} contains {len(sents)} sentences")

		# Save the founded sentences
		os.makedirs(dirname(out_dataset_path), exist_ok=True)
		with open(out_dataset_path, "w") as target_file:
			target_file.writelines(sents)

	def create_parsed_dataset(self, in_dataset_path):
		# Create a parsed dataset from wikipedia
		out_dataset_path = in_dataset_path.replace("/txt/", "/parsed/").replace(".txt", ".parsed")
		self.parse_dataset(in_dataset_path, out_dataset_path, save_as_str=False)

	def create_examples_dataset(self, in_dataset_path):
		# Create a dataset of good example sentences

		# Choose only sentences with at least one NOMLEX predicate
		out_dataset_path = join(dirname(in_dataset_path), "example.txt")
		condition_func = lambda doc: self.args_extractor.extract_arguments(doc, min_arguments=2, return_single=True) != {}
		self.parse_dataset(in_dataset_path, out_dataset_path, condition_func=condition_func, save_as_str=True)

	def create_args_dataset(self, in_dataset_path):
		# Create the ARGS dataset
		out_dataset_path = in_dataset_path.replace("/parsed/", "/datasets/args/").replace(".parsed", "_args.csv")

		# Find the limited verbs for each complement type
		labels_related_verbs = defaultdict(list)
		for arg_type in SYN_ARGS_LABELS + [COMP_PP1, COMP_PP2]:
			labels_related_verbs[arg_type] = self.args_extractor.nom_lexicon.get_verbs_by_arg(arg_type)

		# The total verbs that can appear in the dataset
		# This entry will be updated whenever we sample enough examples of certain labels
		labels_related_verbs["TOTAL"] = list(set(flatten(labels_related_verbs.values())))
		self.create_custom_dataset(in_dataset_path, out_dataset_path, self.get_args_samples, SYN_ARGS_LABELS, labels_related_verbs)

	def create_nouns_dataset(self, in_dataset_path):
		# Create the NOUNS dataset
		out_dataset_path = in_dataset_path.replace("/parsed/", "/datasets/nouns/").replace(".parsed", "_nouns.csv")
		self.create_custom_dataset(in_dataset_path, out_dataset_path, self.get_nouns_samples, SYN_NOUNS_LABELS + [NOM_TYPE_IND_OBJ], None)

	def create_combined_dataset(self, in_dataset_path):
		# Create the ARGS dataset
		out_dataset_path = in_dataset_path.replace("/parsed/", "/datasets/args/").replace(".parsed", "_args.csv")
		self.create_custom_dataset(in_dataset_path, out_dataset_path, self.get_args_samples, SYN_ARGS_LABELS, None)