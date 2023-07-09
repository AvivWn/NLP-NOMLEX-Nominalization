from typing import List, Dict, Any, Optional, Tuple

from demo.handle_specified_tags import tag_extraction, filter_specified_tags, TaggedRanges, \
	seperate_predicate_and_args_ranges
from demo.dynamic_extractions_info import DynamicExtractionsInfo
from yet_another_verb.arguments_extractor.extraction import Extractions, MultiWordExtraction
from yet_another_verb.arguments_extractor.extraction.representation import ParsedOdinMentionRepresentation
from yet_another_verb.arguments_extractor.extraction.utils.extraction_utils import rename_types_to_verbal_active
from yet_another_verb.dependency_parsing import POSTag
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.dependency_parsing.representations import parsed_to_odin


def _filter_events_by_pos(events: List[int], parsed_sent: ParsedText, part_of_speech: POSTag, shift_index: int):
	filtered = []
	for e in events:
		parsed_e = parsed_sent[e - shift_index]
		if part_of_speech in {parsed_e.pos, parsed_e.tag}:
			filtered.append(e)
	return filtered


def _shift_key_idx(thing_by_idx: Dict[int, Any], shift_index: int) -> Dict[int, Any]:
	return {shift_index + idx: thing for idx, thing in thing_by_idx.items()}


def _tag_args_in_extractions(
		parsed_sent: ParsedText, tagged_ranges: Optional[TaggedRanges],
		sent_shift_idx: int, multi_word_extraction: MultiWordExtraction
):
	sentence_range = (sent_shift_idx, sent_shift_idx + len(parsed_sent))
	tagged_ranges = filter_specified_tags(tagged_ranges, sentence_range)

	predicates_indices, arg_tagged_ranges = seperate_predicate_and_args_ranges(tagged_ranges)
	tagged_extractions = tag_extraction(multi_word_extraction, predicates_indices, arg_tagged_ranges)

	return tagged_extractions


def _represent_sentence_extraction(
		parsed_sent: ParsedText, multi_word_extraction: MultiWordExtraction,
		document_id: str, sentence_id: int, sent_shift_idx: int, rename_to_verbal_active=False
):
	if rename_to_verbal_active:
		for ext in multi_word_extraction.extractions:
			rename_types_to_verbal_active(ext)

	mentions_per_idx = ParsedOdinMentionRepresentation(document_id, sentence_id, sent_shift_idx, use_head_idx_only=True). \
		represent_by_word(multi_word_extraction, combined=True)
	mentions_per_idx = _shift_key_idx(mentions_per_idx, sent_shift_idx)

	sorted_events = sorted(mentions_per_idx.keys(), key=lambda e: len(mentions_per_idx[e]), reverse=True)
	sorted_noun_events = _filter_events_by_pos(sorted_events, parsed_sent, POSTag.NOUN, sent_shift_idx)
	sorted_verb_events = _filter_events_by_pos(sorted_events, parsed_sent, POSTag.VERB, sent_shift_idx)

	return mentions_per_idx, sorted_noun_events, sorted_verb_events


def generate_args_extraction_info(
		parsed_sent: ParsedText, multi_word_extraction: MultiWordExtraction, document_id: str,
		tagged_ranges: Optional[TaggedRanges] = None,
		extractions_info: Optional[DynamicExtractionsInfo] = None,
		rename_to_verbal_active=False
) -> Tuple[Extractions, DynamicExtractionsInfo]:
	tagged_ranges = tagged_ranges if tagged_ranges is not None else {}
	extractions_info = DynamicExtractionsInfo() if extractions_info is None else extractions_info

	predicate_indices = multi_word_extraction.extractions_per_idx.keys()

	tagged_extractions = _tag_args_in_extractions(
		parsed_sent, tagged_ranges, extractions_info.sent_shift_idx, multi_word_extraction)
	mentions_by_event, sorted_noun_events, sorted_verb_events = _represent_sentence_extraction(
		parsed_sent, multi_word_extraction, document_id,
		extractions_info.sentence_id, extractions_info.sent_shift_idx, rename_to_verbal_active)

	if True or len(mentions_by_event) > 0:
		extractions_info.parsed_data.update(parsed_to_odin(
			parsed_sent, document_id, extractions_info.parsed_data, predicate_indices))
		extractions_info.mentions_by_event.update(mentions_by_event)
		extractions_info.sorted_noun_events.append(sorted_noun_events)
		extractions_info.sorted_verb_events.append(sorted_verb_events)

		extractions_info.sentence_id += 1
		extractions_info.sent_shift_idx += len(parsed_sent)

	return tagged_extractions, extractions_info
