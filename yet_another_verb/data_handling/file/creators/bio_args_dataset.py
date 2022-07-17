from typing import List, Optional
from collections import namedtuple

import pandas as pd
from tqdm import tqdm

from yet_another_verb.arguments_extractor.extraction import MultiWordExtractions
from yet_another_verb.arguments_extractor.extraction.representation import BIORepresentation
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.dependency_parsing.dependency_parser.parsed_word import ParsedWord
from yet_another_verb.data_handling import PKLFileHandler, CSVFileHandler
from yet_another_verb.data_handling.dataset_creator import DatasetCreator
from yet_another_verb.word_to_verb.verb_translator import VerbTranslator

BIOTaggedSentence = namedtuple("BIOTaggedSentence", "predicate words labels")


class BIOArgsDatasetCreator(DatasetCreator):
	def __init__(
			self, in_dataset_path: str, verb_translator: VerbTranslator,
			dataset_size=None, limited_postags=None, limited_types=None, use_base_verb=False,
			replace_in_sentence=False, avoid_outside_tag=False, tag_predicate=True,
			**kwargs):
		super().__init__(dataset_size)
		self.in_dataset_path = in_dataset_path
		self.verb_translator = verb_translator

		self.limited_postags = limited_postags
		self.limited_types = limited_types
		self.use_base_verb = use_base_verb
		self.replace_in_sentence = replace_in_sentence
		self.avoid_outside_tag = avoid_outside_tag  # avoid O

		self.bio_representation = BIORepresentation(tag_predicate, arg_types=self.limited_types)

	def _generate_bio_tagged_sentence(
			self, words: ParsedText, predicate_idx: int, bio_tags: List[str]) -> Optional[BIOTaggedSentence]:
		predicate = words[predicate_idx]
		predicate_lemma = predicate.lemma
		if self.use_base_verb:
			predicate_lemma = self.verb_translator.translate(predicate_lemma, predicate.pos)

		tokens = words.words
		if self.replace_in_sentence:
			tokens[predicate_idx] = predicate_lemma

		if self.avoid_outside_tag:
			relevant_idxs = [i for i in range(len(bio_tags)) if bio_tags[i] != 'O']
			tokens = [tokens[i] for i in range(len(tokens)) if i in relevant_idxs]
			bio_tags = [bio_tags[i] for i in range(len(bio_tags)) if i in relevant_idxs]

		if len(tokens) == 0 or set(bio_tags) == {'O'}:
			return

		return BIOTaggedSentence(predicate_lemma, " ".join(tokens), " ".join(bio_tags))

	def _should_filter(self, predicate: ParsedWord) -> bool:
		if self.limited_postags is not None:
			if predicate.tag not in self.limited_postags and predicate.pos not in self.limited_postags:
				return True

		if not self.verb_translator.is_transable(predicate.lemma, predicate.pos):
			return True

		return False

	def extractions_to_bio_tags(self, multi_word_extractions: MultiWordExtractions) -> List[BIOTaggedSentence]:
		bios = []

		for multi_word_ext in tqdm(multi_word_extractions, leave=False):
			words = multi_word_ext.words

			if not isinstance(words, ParsedText):
				raise Exception("Cannot generate dataset from unparsed sentence.")

			bio_tags_by_predicate = self.bio_representation.represent_by_word(multi_word_ext, combined=True)

			for predicate_idx, bio_tags in bio_tags_by_predicate.items():
				if not self._should_filter(words[predicate_idx]):
					tagged_sentence = self._generate_bio_tagged_sentence(words, predicate_idx, bio_tags)

					if tagged_sentence is not None:
						bios.append(tagged_sentence)

		return bios

	def create_dataset(self, out_dataset_path: str):
		in_dataset = PKLFileHandler.load(self.in_dataset_path)
		out_dataset = self.extractions_to_bio_tags(in_dataset)
		CSVFileHandler.save(out_dataset_path, pd.DataFrame(out_dataset))
