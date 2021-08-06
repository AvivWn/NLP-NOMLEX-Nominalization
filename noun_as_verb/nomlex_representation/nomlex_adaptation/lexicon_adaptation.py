from copy import deepcopy
from typing import Optional, List

from tqdm import tqdm

from noun_as_verb.nomlex_representation.lexical_entry import LexicalEntry
from noun_as_verb.nomlex_representation.lexicon import Lexicon
from noun_as_verb.nomlex_representation.lexicon_constants import LexiconType, LexiconTag, EntryProperty, EntryType
from noun_as_verb.nomlex_representation.nomlex_adaptation.entry.entry_adaptation import adapt_entry
from noun_as_verb.nomlex_representation.nomlex_adaptation.entry.entry_division import devide_to_entries
from noun_as_verb.nomlex_representation.nomlex_adaptation.entry.entry_omission import should_ommit_entry
from noun_as_verb.nomlex_representation.nomlex_adaptation.entry.entry_simplification import simplify_entry


def add_to_lexicon(lexicon: Lexicon, entry: Optional[LexicalEntry]):
	if entry is None:
		return

	if entry.orth not in lexicon.entries.keys():
		lexicon.entries[entry.orth] = entry
		return

	exist_entry = lexicon.entries[entry.orth]
	exist_entry.related_orths = list(set(exist_entry.related_orths + entry.related_orths))
	for subcat_type, subcat in entry.subcats.items():
		if subcat_type not in exist_entry.subcats:
			exist_entry.subcats[subcat_type] = []

		exist_entry.subcats[subcat_type] += subcat


def create_entry_from_dict(entry: dict, related_orths: List[str]) -> LexicalEntry:
	return LexicalEntry(
		orth=entry[EntryProperty.ORTH],
		related_orths=related_orths,
		subcats=entry[EntryProperty.SUBCATS]
	)


def create_verb_entry(entry: dict) -> Optional[LexicalEntry]:
	if EntryProperty.VERB not in entry:
		return None

	verb_entry = deepcopy(entry)
	adapt_entry(verb_entry, LexiconType.VERB)
	return create_entry_from_dict(verb_entry, [entry[EntryProperty.ORTH]])


def create_nom_entry(entry: dict) -> Optional[LexicalEntry]:
	if EntryType.NOM not in entry[EntryProperty.TYPE]:
		return None

	dict_nom_entry = deepcopy(entry)
	return adapt_entry(dict_nom_entry, LexiconType.NOUN)


def adapt_entry_and_insert(lexicon: Lexicon, entry: dict):
	verb_entry = create_verb_entry(entry)
	add_to_lexicon(lexicon, verb_entry)

	nom_entry = create_nom_entry(entry)
	add_to_lexicon(lexicon, nom_entry)


def generate_adapted_lexicon(lexicon_entries: List[dict]) -> Lexicon:
	adapted_lexicon = Lexicon()

	for raw_entry in tqdm(lexicon_entries, "Adapting the lexicon", leave=False):
		simplify_entry(raw_entry)

		for entry in devide_to_entries(raw_entry):
			if not should_ommit_entry(entry):
				adapt_entry_and_insert(adapted_lexicon, entry)

	return adapted_lexicon
