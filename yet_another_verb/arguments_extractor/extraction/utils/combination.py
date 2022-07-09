from collections import defaultdict
from typing import Tuple, Set

from yet_another_verb.arguments_extractor.extraction.extracted_argument import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.extraction import Extraction, Extractions
from yet_another_verb.arguments_extractor.extraction.multi_word_extraction import MultiWordExtraction


def is_predicate_arg(arg, predicate_idx):
	return predicate_idx in arg.arg_idxs


def separate_args_by_determination(args: Set[ExtractedArgument], predicate_idx: int) -> Tuple[set, set]:
	arg_types_by_idx = defaultdict(set)
	for arg in args:
		for idx in arg.arg_idxs:
			arg_types_by_idx[idx].add(arg.arg_type)

	determined, undetermined = set(), set()
	for arg in args:
		is_this_only_type = all([len(arg_types_by_idx[i]) == 1 for i in arg.arg_idxs])
		is_other_can_be_type = any([arg.arg_type in arg_types_by_idx[j] for j in arg_types_by_idx.keys() if j not in arg.arg_idxs])

		# Argument that refers to a predicate is superior to another argument option
		if is_this_only_type and (is_predicate_arg(arg, predicate_idx) or not is_other_can_be_type):
			determined.add(arg)
		else:
			undetermined.add(arg)

	return determined, undetermined


def combine_extractions(extractions: Extractions, safe_combine=False) -> Extraction:
	if len(extractions) == 0:
		raise Exception("Cannot combine zero extractions.")

	total_args = set()
	for extraction in extractions:
		total_args.update(extraction.args)

	ex = extractions[0]
	words, predicate_idx, predicate_lemma = ex.words, ex.predicate_idx, ex.predicate_lemma

	if safe_combine:
		determined, undetermined = separate_args_by_determination(total_args, predicate_idx)
	else:
		determined, undetermined = total_args, set()

	return Extraction(
		words=words, predicate_idx=predicate_idx, predicate_lemma=predicate_lemma,
		args=determined, undetermined_args=undetermined)


def combine_extractions_by_word(multi_word_ext: MultiWordExtraction, safe_combine=False) -> MultiWordExtraction:
	extractions_per_idx = {}
	for predicate_idx, extractions in multi_word_ext.extractions_per_idx.items():
		if len(extractions) != 0:
			extractions_repr = combine_extractions(extractions, safe_combine)
			extractions_per_idx[predicate_idx] = [extractions_repr]

	return MultiWordExtraction(
		words=multi_word_ext.words,
		extractions_per_idx=extractions_per_idx)
