from typing import List, Dict, Any, Optional, Tuple

from demo.handle_specified_tags import tag_extraction, filter_specified_tags, TaggedRanges, \
	seperate_predicate_and_args_ranges
from demo.dynamic_extractions_info import DynamicExtractionsInfo
from yet_another_verb.arguments_extractor.extraction.comparators.extraction_matcher import ExtractionMatcher
from yet_another_verb.arguments_extractor.extraction import Extractions, MultiWordExtraction
from yet_another_verb.arguments_extractor.extraction.representation import ParsedOdinMentionRepresentation
from yet_another_verb.dependency_parsing import POSTag
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.dependency_parsing.representations import parsed_to_odin
from yet_another_verb import ArgsExtractor


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
		document_id: str, sentence_id: int, sent_shift_idx: int
):
	mentions_per_idx = ParsedOdinMentionRepresentation(document_id, sentence_id, sent_shift_idx). \
		represent_by_word(multi_word_extraction, combined=True)
	mentions_per_idx = _shift_key_idx(mentions_per_idx, sent_shift_idx)

	sorted_events = sorted(mentions_per_idx.keys(), key=lambda e: len(mentions_per_idx[e]), reverse=True)
	sorted_noun_events = _filter_events_by_pos(sorted_events, parsed_sent, POSTag.NOUN, sent_shift_idx)
	sorted_verb_events = _filter_events_by_pos(sorted_events, parsed_sent, POSTag.VERB, sent_shift_idx)

	return mentions_per_idx, sorted_noun_events, sorted_verb_events


def generate_args_extraction_info(
		parsed_text: ParsedText, args_extractor: ArgsExtractor,
		document_id: str, tagged_ranges: Optional[TaggedRanges] = None,
		references: Optional[Extractions] = None, reference_matcher: Optional[ExtractionMatcher] = None,
		dynamic_extractions_info: Optional[DynamicExtractionsInfo] = None
) -> Tuple[Extractions, DynamicExtractionsInfo]:
	tagged_ranges = tagged_ranges if tagged_ranges is not None else {}
	dynamic_extractions_info = DynamicExtractionsInfo() if dynamic_extractions_info is None else dynamic_extractions_info

	tagged_extractions = []

	for parsed_sent in parsed_text.sents:
		parsed_sent = parsed_sent.as_standalone_parsed_text()
		multi_word_extraction = args_extractor.extract_multiword(
			parsed_sent, references=references, reference_matcher=reference_matcher)

		tagged_extractions += _tag_args_in_extractions(parsed_sent, tagged_ranges, dynamic_extractions_info.sent_shift_idx, multi_word_extraction)
		mentions_by_event, sorted_noun_events, sorted_verb_events = _represent_sentence_extraction(
			parsed_sent, multi_word_extraction, document_id,
			dynamic_extractions_info.sentence_id, dynamic_extractions_info.sent_shift_idx)

		if references is None or len(mentions_by_event) > 0:
			dynamic_extractions_info.parsed_data.update(parsed_to_odin(parsed_sent, document_id, dynamic_extractions_info.parsed_data))
			dynamic_extractions_info.mentions_by_event.update(mentions_by_event)
			dynamic_extractions_info.sorted_noun_events.append(sorted_noun_events)
			dynamic_extractions_info.sorted_verb_events.append(sorted_verb_events)

			dynamic_extractions_info.sentence_id += 1
			dynamic_extractions_info.sent_shift_idx += len(parsed_sent)

	return tagged_extractions, dynamic_extractions_info
