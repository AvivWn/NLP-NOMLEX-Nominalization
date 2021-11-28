from copy import deepcopy
from typing import Optional, List
from collections import defaultdict
from itertools import chain

from tqdm import tqdm

from yet_another_verb.nomlex.representation.lexical_entry import LexicalEntry
from yet_another_verb.nomlex.representation.lexicon import Lexicon
from yet_another_verb.nomlex.constants import LexiconType, EntryProperty, EntryType, LexiconTag
from yet_another_verb.nomlex.adaptation.entry.entry_adaptation import adapt_entry
from yet_another_verb.nomlex.adaptation.entry.entry_division import devide_to_entries
from yet_another_verb.nomlex.adaptation.entry.entry_omission import should_ommit_entry
from yet_another_verb.nomlex.adaptation.entry.entry_simplification import simplify_entry


def add_to_lexicon(lexicon: Lexicon, entry: Optional[LexicalEntry]):
	if entry is None:
		return

	if entry.orth not in lexicon.entries.keys():
		lexicon.entries[entry.orth] = []

	lexicon.entries[entry.orth].append(entry)


def create_entry_from_dict(entry: dict, related_orths: List[str], ambiguous_forms: List[str]) -> LexicalEntry:
	return LexicalEntry(
		orth=entry[EntryProperty.ORTH],
		related_orths=related_orths,
		ambiguous_forms=ambiguous_forms,
		subcats=entry[EntryProperty.SUBCATS]
	)


def create_verb_entry(entry: dict) -> Optional[LexicalEntry]:
	if EntryProperty.VERB not in entry:
		return None

	verb_entry = deepcopy(entry)
	adapt_entry(verb_entry, LexiconType.VERB)
	return create_entry_from_dict(verb_entry, [entry[EntryProperty.ORTH]], [])


def create_nom_entry(entry: dict) -> Optional[LexicalEntry]:
	if entry[EntryProperty.TYPE] not in [EntryType.NOM]:
		return None

	nom_entry = deepcopy(entry)
	adapt_entry(nom_entry, LexiconType.NOUN)

	ambiguous_forms = []
	is_noun_exist = set(nom_entry[EntryProperty.NOUN]).issubset([LexiconTag.EXISTS, LexiconTag.RARE_NOM, LexiconTag.RARE_NOUN])
	is_singular_noun_exist = is_noun_exist or LexiconTag.SING_ONLY in nom_entry[EntryProperty.NOUN]
	is_plural_noun_exist = is_noun_exist or LexiconTag.PLUR_ONLY in nom_entry[EntryProperty.NOUN]

	if is_singular_noun_exist or nom_entry[EntryProperty.HAS_ANOTHER_ENTRY]:
		ambiguous_forms.append(nom_entry[EntryProperty.ORTH])

	if nom_entry[EntryProperty.PLURAL] is not None and (is_plural_noun_exist or nom_entry[EntryProperty.HAS_ANOTHER_ENTRY]):
		ambiguous_forms.append(nom_entry[EntryProperty.PLURAL])

	return create_entry_from_dict(nom_entry, [entry[EntryProperty.VERB]], ambiguous_forms)


def adapt_entry_and_insert(lexicon: Lexicon, entry: dict):
	verb_entry = create_verb_entry(entry)
	add_to_lexicon(lexicon, verb_entry)

	nom_entry = create_nom_entry(entry)
	add_to_lexicon(lexicon, nom_entry)


def update_related_orths(lexicon: Lexicon):
	orths_by_orth = defaultdict(set)
	for entry in list(chain(*lexicon.entries.values())):
		orths_by_orth[entry.orth].update(entry.related_orths)

	for entry in list(chain(*lexicon.entries.values())):
		new_related_orths = list(set(chain(*[orths_by_orth[orth] for orth in [entry.orth] + entry.related_orths])))
		new_related_orths.remove(entry.orth)
		entry.related_orths = list(set(entry.related_orths + new_related_orths))


def generate_adapted_lexicon(lexicon_entries: List[dict]) -> Lexicon:
	adapted_lexicon = Lexicon()

	orth_entry_count = defaultdict(int)
	for raw_entry in lexicon_entries:
		simplify_entry(raw_entry)
		orth_entry_count[(raw_entry[EntryProperty.ORTH], raw_entry[EntryProperty.TYPE])] += 1

	for raw_entry in tqdm(lexicon_entries, "Adapting the lexicon", leave=False):
		raw_entry[EntryProperty.HAS_ANOTHER_ENTRY] = orth_entry_count[
			(raw_entry[EntryProperty.ORTH], raw_entry[EntryProperty.TYPE])] > 1

		for entry in devide_to_entries(raw_entry):
			if not should_ommit_entry(entry):
				adapt_entry_and_insert(adapted_lexicon, entry)

	update_related_orths(adapted_lexicon)
	return adapted_lexicon
