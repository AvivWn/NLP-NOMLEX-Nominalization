from typing import List, Optional

from yet_another_verb.nomlex.constants import SubcatType, EntryProperty
from yet_another_verb.arguments_extractor.extraction import ArgumentType

ARG_PLURAL = {
	ArgumentType.SUBJ: [SubcatType.NOM_INTRANS_RECIP]
}

ARG_SUBJUNCT = {
	ArgumentType.SBAR: [
		SubcatType.NOM_S_SUBJUNCT,
		SubcatType.NOM_PP_THAT_S_SUBJUNCT,
		SubcatType.NOM_NP_AS_IF_S_SUBJUNCT
	]
}

ARG_CONTROLLED = {
	ArgumentType.MODIFIER: {
		SubcatType.NOM_NP_AS_ADJP: [ArgumentType.OBJ],
	},
	ArgumentType.PP: {
		SubcatType.NOM_AS_NP: [ArgumentType.SUBJ],
		SubcatType.NOM_PART_AS_NP: [ArgumentType.SUBJ],
		SubcatType.NOM_NP_AS_NP: [ArgumentType.OBJ],
		SubcatType.NOM_PART_NP_AS_NP: [ArgumentType.OBJ],
		SubcatType.NOM_NP_AS_NP_SC: [ArgumentType.SUBJ],
		SubcatType.NOM_NP_PP_AS_NP: [ArgumentType.OBJ]
	},
	ArgumentType.ING: {
		SubcatType.NOM_NP_AS_ING: [ArgumentType.OBJ],
		SubcatType.NOM_NP_ING: [ArgumentType.NP],
		SubcatType.NOM_NP_ING_OC: [ArgumentType.OBJ],
		SubcatType.NOM_NP_ING_SC: [ArgumentType.SUBJ],
		SubcatType.NOM_ING_SC: [ArgumentType.SUBJ],
		SubcatType.NOM_PART_ING_SC: [ArgumentType.SUBJ],
		SubcatType.NOM_P_NP_ING: [ArgumentType.PP],
		SubcatType.NOM_NP_P_NP_ING: [ArgumentType.PP],
		SubcatType.NOM_NP_P_ING: [ArgumentType.NP],
		SubcatType.NOM_NP_P_ING_OC: [ArgumentType.OBJ],
		SubcatType.NOM_NP_P_ING_SC: [ArgumentType.SUBJ],
		SubcatType.NOM_P_ING_SC: [ArgumentType.SUBJ],
	},
	ArgumentType.TO_INF: {
		SubcatType.NOM_NP_TO_INF_OC: [ArgumentType.OBJ],
		SubcatType.NOM_NP_TO_INF_SC: [ArgumentType.SUBJ],
		SubcatType.NOM_NP_TO_INF_VC: [ArgumentType.SUBJ, ArgumentType.OBJ],
		SubcatType.NOM_TO_INF_SC: [ArgumentType.SUBJ],
		SubcatType.NOM_P_NP_TO_INF_OC: [ArgumentType.PP],
		SubcatType.NOM_P_NP_TO_INF: [ArgumentType.PP],
		SubcatType.NOM_P_NP_TO_INF_VC: [ArgumentType.SUBJ, ArgumentType.PP]
	}
}

ARG_ATTRIBUTES_PROPERTY = {
	ArgumentType.SUBJ: EntryProperty.SUBJ_ATTRIBUTE,
	ArgumentType.OBJ: EntryProperty.OBJ_ATTRIBUTE,
	ArgumentType.IND_OBJ: EntryProperty.IND_OBJ_ATTRIBUTE
}

#
# ARG_CONTIGUOUS_TO = {
# 	ArgumentType.OBJ: {
# 		SubcatType.NOM_NP_NP: {
# 			LexiconType.VERB: None,
# 			LexiconType.NOUN: ArgumentType.IND_OBJ
# 		}
# 	},
# 	ArgumentType.TO_INF: {
# 		SubcatType.NOM_NP_TO_INF_VC: {
# 			LexiconType.VERB: None,
# 			LexiconType.NOUN: ArgumentType.OBJ
# 		},
# 		SubcatType.NOM_P_NP_TO_INF: {
# 			LexiconType.VERB: None,
# 			LexiconType.NOUN: ArgumentType.PP
# 		},
# 		SubcatType.NOM_P_NP_TO_INF_VC: {
# 			LexiconType.VERB: ArgumentType.PP,
# 			LexiconType.NOUN: ArgumentType.PP
# 		},
# 	},
# 	ArgumentType.ING: {
# 		SubcatType.NOM_NP_AS_ING: {
# 			LexiconType.VERB: ArgumentType.OBJ,
# 			LexiconType.NOUN: ArgumentType.OBJ
# 		},
# 	},
# 	SubcatType.ING: {
# 		SubcatType.NOM_NP_ING: {
# 			LexiconType.VERB: ArgumentType.NP,
# 			LexiconType.NOUN: ArgumentType.NP
# 		},
# 		SubcatType.NOM_NP_ING_OC: {
# 			LexiconType.VERB: ArgumentType.OBJ,
# 			LexiconType.NOUN: ArgumentType.OBJ
# 		},
# 		SubcatType.NOM_NP_ING_SC: {
# 			LexiconType.VERB: ArgumentType.OBJ,
# 			LexiconType.NOUN: ArgumentType.OBJ
# 		}
# 	}
# }


def get_plural_property(subcat_type: SubcatType, arg_type: ArgumentType) -> bool:
	if arg_type not in ARG_PLURAL:
		return False

	return subcat_type in ARG_PLURAL[arg_type]


def get_subjunct_property(subcat_type: SubcatType, arg_type: ArgumentType) -> bool:
	if arg_type not in ARG_SUBJUNCT:
		return False

	return subcat_type in ARG_SUBJUNCT[arg_type]


def get_controlled_args(subcat_type: SubcatType, arg_type: ArgumentType) -> List[ArgumentType]:
	return ARG_CONTROLLED.get(arg_type, {}).get(subcat_type, [])


def get_arg_attributes_property(arg_type: ArgumentType) -> Optional[EntryProperty]:
	return ARG_ATTRIBUTES_PROPERTY.get(arg_type, None)
