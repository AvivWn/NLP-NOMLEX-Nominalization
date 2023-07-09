from collections import defaultdict
from itertools import product
from typing import List, Dict, Optional

import numpy as np

from yet_another_verb.arguments_extractor.extraction import Extractions, Extraction
from yet_another_verb.arguments_extractor.extraction.utils.argument_utils import specify_pp_type_in_arg, \
	remove_pp_type_in_arg, reduce_args_by_arg_types, rename_type_to_verbal_active
from yet_another_verb.arguments_extractor.extraction.utils.combination import separate_args_by_determination
from yet_another_verb.arguments_extractor.extraction.utils.reconstruction import reconstruct_extraction
from yet_another_verb.word_to_verb.verb_translator import VerbTranslator


def aggregate_by_predicate(
		extractions: Extractions, verb_translator: Optional[VerbTranslator] = None) -> Dict[str, Extractions]:
	extractions_by_verb = defaultdict(list)
	for ext in extractions:
		predicate = ext.predicate_lemma
		if verb_translator is not None:
			predicate = verb_translator.translate(ext.predicate_lemma, ext.predicate_postag)

		extractions_by_verb[predicate].append(ext)

	return extractions_by_verb


def get_arg_types_in_extractions(extractions: Extractions) -> List[str]:
	arg_types = set()

	for ext in extractions:
		arg_types.update(ext.arg_types)

	return list(arg_types)


def modify_pp_in_args(extraction, consider_pp_type):
	for arg in extraction.all_args:
		if consider_pp_type:
			specify_pp_type_in_arg(extraction.words, arg)
		else:
			remove_pp_type_in_arg(arg)


def get_extractions_with_modified_pp(extractions, consider_pp_type, reconstruct=True) -> Extractions:
	new_extractions = []

	for extraction in extractions:
		new_extraction = reconstruct_extraction(extraction) if reconstruct else extraction
		modify_pp_in_args(new_extraction, consider_pp_type)
		new_extractions.append(new_extraction)

	return new_extractions


def separate_extractions_with_repeated_arg_type(extractions: Extractions) -> Extractions:
	separated_extractions = []

	for ext in extractions:
		args_by_type = defaultdict(list)
		for arg in ext.args:
			args_by_type[arg.arg_type].append(arg)

		for args_comb in product(*args_by_type.values()):
			new_ext = reconstruct_extraction(ext, args=args_comb)
			separated_extractions.append(new_ext)

	return separated_extractions


def reduce_extractions_by_arg_types(
		extractions: Extractions, arg_types=None, keep_undetermined=False, reconstruct=True) -> Extractions:
	relevant_exts = []
	for extraction in extractions:
		args = extraction.args
		undetermined_args = extraction.undetermined_args

		if arg_types is not None:
			args = reduce_args_by_arg_types(args, arg_types)

			if keep_undetermined:
				undetermined_args = reduce_args_by_arg_types(undetermined_args, arg_types)
				_, undetermined_args = separate_args_by_determination(undetermined_args, extraction.predicate_idx)
			else:
				undetermined_args = set()

		if len(args) > 0:
			if reconstruct:
				extraction = reconstruct_extraction(extraction, args=args, undetermined_args=undetermined_args)
			else:
				extraction.args = args
				extraction.undetermined_args = undetermined_args

			relevant_exts.append(extraction)

	return relevant_exts


def convert_encodings_to_numpy(extractions: Extractions):
	for ext in extractions:
		for arg in ext.all_args:
			arg.encoding = np.array(arg.encoding)


def rename_types_to_verbal_active(extraction: Extraction):
	for arg in extraction.args:
		rename_type_to_verbal_active(arg)
