import re
from typing import Dict, Tuple, List

from yet_another_verb.arguments_extractor.extraction import Extractions, MultiWordExtraction
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText

TAG_GRP = "tag"
VALUE_GRP = "value"
TAGS_PATTERN = fr"\[(?P<{TAG_GRP}>[^\[\] ]*) (?P<{VALUE_GRP}>[^\[\]]*)\]"
PREDICATE_TAG = "#"

Range = Tuple[int, int]
Ranges = List[Range]
TaggedRanges = Dict[Range, str]


def parse_specified_tags(text: str) -> Tuple[str, TaggedRanges]:
	matches = re.finditer(TAGS_PATTERN, text)

	tag_idx_shift = 0
	tagged_ranges = {}
	for i, match in enumerate(matches):
		tag = match.group(TAG_GRP)
		start, end = match.span(VALUE_GRP)

		# remove tags from text
		text = text[0:max(match.span()[0] - tag_idx_shift, 0)] + \
			match.group(VALUE_GRP) + text[match.span()[1] - tag_idx_shift:]

		tag_idx_shift += 1 + len(tag) + 1  # [TAG<space>
		tagged_range = (start - tag_idx_shift, end - tag_idx_shift)
		tagged_ranges[tagged_range] = tag
		tag_idx_shift += 1  # ]

	return text, tagged_ranges


def filter_specified_tags(tagged_ranges: TaggedRanges, limited_range: Range) -> TaggedRanges:
	return {r: tag for r, tag in tagged_ranges.items() if r[0] >= limited_range[0] and r[1] <= limited_range[1]}


def translate_char_ranges_to_word_ranges(parsed_text: ParsedText, tagged_char_ranges: TaggedRanges) -> TaggedRanges:
	word_idx, original_char_idx = 0, 0
	original_text, tokenized_text = parsed_text.text, parsed_text.tokenized_text
	char_idx_to_word_idx_map = {}

	for char_idx in range(len(tokenized_text)):
		char_idx_to_word_idx_map[original_char_idx] = word_idx

		if tokenized_text[char_idx] == ' ':
			word_idx += 1

		if original_text[original_char_idx] == tokenized_text[char_idx]:
			original_char_idx += 1

	tagged_word_ranges = {}
	for char_range, tag in tagged_char_ranges.items():
		start_char_idx, end_char_idx = char_range
		words_range = (char_idx_to_word_idx_map[start_char_idx], char_idx_to_word_idx_map[end_char_idx-1])
		tagged_word_ranges[words_range] = tag

	return tagged_word_ranges


def seperate_predicate_and_args_ranges(tagged_ranges: TaggedRanges) -> Tuple[List[int], TaggedRanges]:
	arg_tagged_ranges = {}
	predicates_indices = []

	for tagged_range, tag in tagged_ranges.items():
		if tag == PREDICATE_TAG:
			# Predicate should contain a single word
			if tagged_range[0] == tagged_range[1]:
				predicates_indices.append(tagged_range[0])
		else:
			arg_tagged_ranges[tagged_range] = tag

	return predicates_indices, arg_tagged_ranges


def tag_extraction(
		multi_word_extraction: MultiWordExtraction,
		predicates_indices: List[int], arg_tagged_ranges: TaggedRanges
) -> Extractions:
	if len(predicates_indices) == 0:
		return []

	# There should be at most one predicate
	predicate_idx = predicates_indices[0]

	# The tagged predicate might not yield any extractions
	extractions = multi_word_extraction.extractions_per_idx.get(predicate_idx, [])

	for extraction in extractions:
		for arg_range, tag in arg_tagged_ranges.items():
			extraction.tag_arg_by_range(arg_range, tag)

	return extractions
