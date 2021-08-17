from copy import deepcopy

from yet_another_verb.nomlex.constants import LexiconType, LexiconTag, EntryProperty
from yet_another_verb.nomlex.constants import SubcatType, SubcatProperty
from yet_another_verb.nomlex.constants.entry_property import ENTRY_PROPERTIES
from yet_another_verb.nomlex.adaptation.modifications import get_correct_subcat_type
from yet_another_verb.nomlex.adaptation.subcat.subcat_adaptation import adapt_subcat
from yet_another_verb.nomlex.adaptation.subcat.subcat_omission import should_ommit_subcat
from yet_another_verb.utils.linguistic_utils import in_plural


def remove_irrelevant_properties(entry: dict):
	irrelevant_properties = [
		EntryProperty.NOUN_SUBC, EntryProperty.VERB_SUBC, EntryProperty.FEATURES, EntryProperty.VERB_SUBJ,
		EntryProperty.DET_POSS_NO_OTHER_OBJ, EntryProperty.N_N_MOD_NO_OTHER_OBJ,
		EntryProperty.SUBJ_ATTRIBUTE, EntryProperty.OBJ_ATTRIBUTE, EntryProperty.IND_OBJ_ATTRIBUTE,
		EntryProperty.PLURAL_FREQ, EntryProperty.SINGULAR_FALSE,
		EntryProperty.NOM_TYPE, EntryProperty.TYPE, EntryProperty.SEMI_AUTOMATIC,
		EntryProperty.NOUN, EntryProperty.PLURAL
	]

	for p in entry.keys():
		if p not in ENTRY_PROPERTIES:
			irrelevant_properties.append(p)

	for p in irrelevant_properties:
		entry.pop(p, None)


def specify_properties_in_subcats(entry):
	nom_features = entry.get(EntryProperty.FEATURES, {}).keys()
	subcats = entry.get(EntryProperty.SUBCATS, {}).values()
	any_alternates_appear = any([SubcatProperty.ALTERNATES in subcat.keys() for subcat in subcats])
	subcat_types = deepcopy(entry)[EntryProperty.SUBCATS].keys()

	for subcat_type in subcat_types:
		subcat_info = entry[EntryProperty.SUBCATS][subcat_type]
		subcat_info[SubcatProperty.DET_POSS_NO_OTHER_OBJ] = entry[EntryProperty.DET_POSS_NO_OTHER_OBJ]
		subcat_info[SubcatProperty.N_N_MOD_NO_OTHER_OBJ] = entry[EntryProperty.N_N_MOD_NO_OTHER_OBJ]

		# Add the ALTERNATES constraint for suitable subcats if the entry don't contain any ALTERNATES tag
		if not any_alternates_appear:
			if LexiconTag.SUBJ_OBJ_ALT in nom_features and SubcatType.is_intransitive(subcat_type):
				subcat_info[SubcatProperty.ALTERNATES] = "T"

			if LexiconTag.SUBJ_IND_OBJ_ALT in nom_features and SubcatType.is_transitive(subcat_type):
				subcat_info[SubcatProperty.ALTERNATES] = "T"


def rearrange_subcats(entry: dict, lexicon_type: LexiconType):
	if lexicon_type == LexiconType.NOUN:
		extra_subcats = entry[EntryProperty.NOUN_SUBC]
		for subcat_type, subcat in deepcopy(extra_subcats).items():
			entry[EntryProperty.SUBCATS][subcat_type.replace("NOUN", "NOM")] = subcat

	specify_properties_in_subcats(entry)

	subcats = entry.get(EntryProperty.SUBCATS, {})
	for subcat_type, subcat in deepcopy(subcats).items():
		correct_subcat_type = get_correct_subcat_type(subcat_type)

		if correct_subcat_type != subcat_type:
			subcats[correct_subcat_type] = subcats.pop(subcat_type)
			subcat_type = correct_subcat_type

		if should_ommit_subcat(entry, subcat, subcat_type, lexicon_type):
			subcats.pop(subcat_type)
			continue

		subcats[SubcatType(subcat_type)] = subcats.pop(subcat_type)
		adapt_subcat(entry, SubcatType(subcat_type), lexicon_type)


def adapt_entry(entry: dict, lexicon_type: LexiconType):
	assert len(entry[EntryProperty.NOM_TYPE]) == 1
	entry[EntryProperty.NOM_TYPE] = entry[EntryProperty.NOM_TYPE][0]

	if lexicon_type == LexiconType.VERB:
		entry[EntryProperty.ORTH] = entry[EntryProperty.VERB]
	elif lexicon_type == LexiconType.NOUN and EntryProperty.PLURAL not in entry:
		entry[EntryProperty.PLURAL] = in_plural(entry[EntryProperty.ORTH])

	rearrange_subcats(entry, lexicon_type)
	remove_irrelevant_properties(entry)
