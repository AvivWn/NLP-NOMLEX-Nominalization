from noun_as_verb.nomlex_representation.lexicon_constants import LexiconType, NomTypeProperty, EntryProperty
from noun_as_verb.nomlex_representation.lexicon_constants.subcat_type import SUBCAT_TYPES


def is_supported_subcat_type(subcat_type: str) -> bool:
	return subcat_type in SUBCAT_TYPES


def is_particle_compatible(entry: dict, subcat: dict, lexicon_type: LexiconType) -> bool:
	nom_type = entry[EntryProperty.NOM_TYPE][NomTypeProperty.TYPE]
	is_part_typed_nom = "PART" in nom_type
	is_part_typed_subcat = "PART" in subcat

	if lexicon_type == LexiconType.VERB:
		return True

	return is_part_typed_nom or (not is_part_typed_nom and not is_part_typed_subcat)


def should_ommit_subcat(entry, subcat: dict, subcat_type: str, lexicon_type: LexiconType) -> bool:
	if not is_supported_subcat_type(subcat_type):
		return True

	if not is_particle_compatible(entry, subcat, lexicon_type):
		return True

	return False
