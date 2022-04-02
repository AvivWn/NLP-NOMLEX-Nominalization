from yet_another_verb.nomlex.constants.entry_type import ENTRY_TYPES
from yet_another_verb.nomlex.constants import LexiconTag, NomTypeProperty, EntryProperty
from yet_another_verb.nomlex.constants import SubcatProperty


def is_supported_entry_type(entry: dict) -> bool:
	return entry[EntryProperty.TYPE] in ENTRY_TYPES


def is_include_required_properties(entry: dict) -> bool:
	required_properties = [EntryProperty.ORTH, EntryProperty.NOM_TYPE, EntryProperty.VERB]
	return all(p in entry.keys() for p in required_properties)


def is_orth_compatible_with_particle(entry: dict) -> bool:
	orth = entry[EntryProperty.ORTH]
	particle = entry[EntryProperty.NOM_TYPE][0][NomTypeProperty.PART]

	if particle is None:
		return True

	return orth.startswith(particle) or orth.endswith(particle)


def has_ilegal_alternation(entry: dict) -> bool:
	nom_features = entry.get(EntryProperty.FEATURES, {}).keys()
	nom_subcats = entry.get(EntryProperty.SUBCATS, {}).values()

	# ALTERNATES tag goes together with one of the features SUBJ-OBJ-ALT or SUBJ-IND-OBJ-ALT
	# Check whether or not these features and the ALTERNATES tag appears for this nominalization
	alt_feature_appear = LexiconTag.SUBJ_OBJ_ALT in nom_features or LexiconTag.SUBJ_IND_OBJ_ALT in nom_features
	any_alternates_appear = any([SubcatProperty.ALTERNATES in subcat for subcat in nom_subcats])

	# Assuming that nominalizations with ALTERNATES tags and without any of these features aren't correct
	return not alt_feature_appear and any_alternates_appear


def does_suitable_noun_exist(entry: dict) -> bool:
	# This omission criterion was replaced by "ambiguity" property in the new representation
	return not {
		LexiconTag.EXISTS,
		LexiconTag.RARE_NOM,
		LexiconTag.RARE_NOUN
	}.isdisjoint(entry[EntryProperty.NOUN])


def is_singular_false(entry: dict) -> bool:
	# This omission criterion was replaced by "ambiguity" property in the new representation
	return entry.get(EntryProperty.SINGULAR_FALSE, False)


def has_at_most_single_plural(entry: dict) -> bool:
	plural_form = entry.get(EntryProperty.PLURAL, None)
	return plural_form is None or (isinstance(plural_form, str) and plural_form)


def is_defined_with_multiple_words(entry: dict) -> bool:
	if EntryProperty.ORTH in entry and len(entry[EntryProperty.ORTH].split()) > 1:
		return True

	if EntryProperty.VERB in entry and len(entry[EntryProperty.VERB].split()) > 1:
		return True

	return False


def should_ommit_entry(entry: dict) -> bool:
	if not is_supported_entry_type(entry):
		return True

	if not is_include_required_properties(entry):
		return True

	if not is_orth_compatible_with_particle(entry):
		return True

	if has_ilegal_alternation(entry):
		return True

	if is_singular_false(entry):
		return True

	if not has_at_most_single_plural(entry):
		return True

	if entry[EntryProperty.ORTH] in ["being"]:
		return True

	if is_defined_with_multiple_words(entry):
		return True

	return False
