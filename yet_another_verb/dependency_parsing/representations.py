from typing import Optional

from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText

MENTIONS_KEY = "mentions"
DOCUMENTS_KEY = "documents"
DOCUMENT_ID_KEY = "id"
DOCUMENT_TEXT_KEY = "text"
DOCUMENT_SENTENCES_KEY = "sentences"


def parsed_to_odin(
		parsed_sent: ParsedText, document_id: str, odin_repr: Optional[dict] = None,
		predicate_indices=None) -> dict:
	total_words = []

	sent_root = parsed_sent[:].root

	words, tags, graph_edges = [], [], []
	for parsed_word in parsed_sent:
		words.append(parsed_word.text)
		tags.append(parsed_word.tag)

		if parsed_word.i != sent_root.i:
			if predicate_indices is None or parsed_word.head.i in predicate_indices:
				graph_edges.append({
					"source": parsed_word.head.i,
					"destination": parsed_word.i,
					"relation": parsed_word.dep
				})

	total_words += words
	odin_sentence_info = {
		"words": words,
		"tags": tags,
		"graphs": {
			"universal-basic": {
				"edges": graph_edges,
				"roots": [sent_root.i]
			}
		}
	}

	if odin_repr is None:
		odin_repr = {}

	if DOCUMENTS_KEY not in odin_repr:
		odin_repr[DOCUMENTS_KEY] = {}

	if MENTIONS_KEY not in odin_repr:
		odin_repr[MENTIONS_KEY] = []

	if document_id not in odin_repr[DOCUMENTS_KEY]:
		odin_repr[DOCUMENTS_KEY][document_id] = {
			DOCUMENT_ID_KEY: document_id,
			DOCUMENT_TEXT_KEY: "",
			DOCUMENT_SENTENCES_KEY: []
		}
	else:
		odin_repr[DOCUMENTS_KEY][document_id][DOCUMENT_TEXT_KEY] += " "

	document_repr = odin_repr[DOCUMENTS_KEY][document_id]
	document_repr[DOCUMENT_TEXT_KEY] += " ".join(total_words)
	document_repr[DOCUMENT_SENTENCES_KEY].append(odin_sentence_info)

	return odin_repr
