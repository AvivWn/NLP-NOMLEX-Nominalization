from typing import List

from yet_another_verb.nomlex.constants import LexiconType, SubcatType, ArgumentType

SUBCAT_REQUIRED_ARGS = {
	SubcatType.NOM_PART: {
		LexiconType.VERB: [ArgumentType.PART],
		LexiconType.NOUN: [ArgumentType.PART]
	},
	SubcatType.NOM_NP: {
		LexiconType.VERB: [ArgumentType.OBJ],
		LexiconType.NOUN: []
	},
	SubcatType.NOM_PART_NP: {
		LexiconType.VERB: [ArgumentType.PART, ArgumentType.OBJ],
		LexiconType.NOUN: [ArgumentType.PART]
	},
	SubcatType.NOM_NP_NP: {
		LexiconType.VERB: [ArgumentType.IND_OBJ, ArgumentType.OBJ],
		LexiconType.NOUN: [ArgumentType.IND_OBJ, ArgumentType.OBJ]
	},
	SubcatType.NOM_PP: {
		LexiconType.VERB: [ArgumentType.PP],
		LexiconType.NOUN: [ArgumentType.PP]
	},
	SubcatType.NOM_PART_PP: {
		LexiconType.VERB: [ArgumentType.PART, ArgumentType.PP],
		LexiconType.NOUN: [ArgumentType.PART, ArgumentType.PP]
	},
	SubcatType.NOM_PP_PP: {
		LexiconType.VERB: [ArgumentType.PP1, ArgumentType.PP2],
		LexiconType.NOUN: [ArgumentType.PP1, ArgumentType.PP2]
	},
	SubcatType.NOM_NP_PP: {
		LexiconType.VERB: [ArgumentType.OBJ, ArgumentType.PP],
		LexiconType.NOUN: [ArgumentType.PP]
	},
	SubcatType.NOM_PART_NP_PP: {
		LexiconType.VERB: [ArgumentType.PART, ArgumentType.OBJ, ArgumentType.PP],
		LexiconType.NOUN: [ArgumentType.PART, ArgumentType.PP]
	},
	SubcatType.NOM_NP_TO_NP: {
		LexiconType.VERB: [ArgumentType.OBJ, ArgumentType.IND_OBJ],
		LexiconType.NOUN: [ArgumentType.OBJ, ArgumentType.IND_OBJ]
	},
	SubcatType.NOM_NP_FOR_NP: {
		LexiconType.VERB: [ArgumentType.OBJ, ArgumentType.IND_OBJ],
		LexiconType.NOUN: [ArgumentType.OBJ, ArgumentType.IND_OBJ]
	},
	SubcatType.NOM_NP_PP_PP: {
		LexiconType.VERB: [ArgumentType.OBJ, ArgumentType.PP1, ArgumentType.PP2],
		LexiconType.NOUN: [ArgumentType.PP1, ArgumentType.PP2]
	},
	SubcatType.NOM_ADVP: {
		LexiconType.VERB: [ArgumentType.MODIFIER],
		LexiconType.NOUN: [ArgumentType.MODIFIER]
	},
	SubcatType.NOM_NP_ADVP: {
		LexiconType.VERB: [ArgumentType.OBJ, ArgumentType.MODIFIER],
		LexiconType.NOUN: [ArgumentType.MODIFIER]
	},
	SubcatType.NOM_ADVP_PP: {
		LexiconType.VERB: [ArgumentType.MODIFIER, ArgumentType.PP],
		LexiconType.NOUN: [ArgumentType.MODIFIER, ArgumentType.PP]
	},
	SubcatType.NOM_NP_AS_ING: {
		LexiconType.VERB: [ArgumentType.OBJ, ArgumentType.ING],
		LexiconType.NOUN: [ArgumentType.OBJ, ArgumentType.ING]
	},
	SubcatType.NOM_NP_AS_ADJP: {
		LexiconType.VERB: [ArgumentType.OBJ, ArgumentType.MODIFIER],
		LexiconType.NOUN: [ArgumentType.OBJ, ArgumentType.MODIFIER]
	},
	SubcatType.NOM_AS_NP: {
		LexiconType.VERB: [ArgumentType.SUBJ, ArgumentType.PP],
		LexiconType.NOUN: [ArgumentType.SUBJ, ArgumentType.PP]
	},
	SubcatType.NOM_PART_AS_NP: {
		LexiconType.VERB: [ArgumentType.PART, ArgumentType.SUBJ, ArgumentType.PP],
		LexiconType.NOUN: [ArgumentType.PART, ArgumentType.SUBJ, ArgumentType.PP]
	},
	SubcatType.NOM_NP_AS_NP: {
		LexiconType.VERB: [ArgumentType.OBJ, ArgumentType.PP],
		LexiconType.NOUN: [ArgumentType.OBJ, ArgumentType.PP]
	},
	SubcatType.NOM_PART_NP_AS_NP: {
		LexiconType.VERB: [ArgumentType.PART, ArgumentType.OBJ, ArgumentType.PP],
		LexiconType.NOUN: [ArgumentType.PART, ArgumentType.OBJ, ArgumentType.PP]
	},
	SubcatType.NOM_NP_AS_NP_SC: {
		LexiconType.VERB: [ArgumentType.SUBJ, ArgumentType.OBJ, ArgumentType.PP],
		LexiconType.NOUN: [ArgumentType.SUBJ, ArgumentType.PP]
	},
	SubcatType.NOM_NP_PP_AS_NP: {
		LexiconType.VERB: [ArgumentType.OBJ, ArgumentType.PP, ArgumentType.PP],
		LexiconType.NOUN: [ArgumentType.OBJ, ArgumentType.PP, ArgumentType.PP]
	},
	SubcatType.NOM_ING_SC: {
		LexiconType.VERB: [ArgumentType.ING],
		LexiconType.NOUN: [ArgumentType.ING]
	},
	SubcatType.NOM_PART_ING_SC: {
		LexiconType.VERB: [ArgumentType.PART, ArgumentType.ING],
		LexiconType.NOUN: [ArgumentType.PART, ArgumentType.ING]
	},
	SubcatType.NOM_NP_ING: {
		LexiconType.VERB: [ArgumentType.NP, ArgumentType.ING],
		LexiconType.NOUN: [ArgumentType.NP, ArgumentType.ING]
	},
	SubcatType.NOM_NP_ING_OC: {
		LexiconType.VERB: [ArgumentType.OBJ, ArgumentType.ING],
		LexiconType.NOUN: [ArgumentType.OBJ, ArgumentType.ING]
	},
	SubcatType.NOM_NP_ING_SC: {
		LexiconType.VERB: [ArgumentType.OBJ, ArgumentType.ING],
		LexiconType.NOUN: [ArgumentType.OBJ, ArgumentType.ING]
	},
	SubcatType.NOM_P_ING_SC: {
		LexiconType.VERB: [ArgumentType.ING],
		LexiconType.NOUN: [ArgumentType.ING]
	},
	SubcatType.NOM_NP_P_ING: {
		LexiconType.VERB: [ArgumentType.NP, ArgumentType.ING],
		LexiconType.NOUN: [ArgumentType.NP, ArgumentType.ING]
	},
	SubcatType.NOM_NP_P_ING_OC: {
		LexiconType.VERB: [ArgumentType.OBJ, ArgumentType.ING],
		LexiconType.NOUN: [ArgumentType.OBJ, ArgumentType.ING]
	},
	SubcatType.NOM_NP_P_ING_SC: {
		LexiconType.VERB: [ArgumentType.OBJ, ArgumentType.ING],
		LexiconType.NOUN: [ArgumentType.OBJ, ArgumentType.ING]
	},
	SubcatType.NOM_P_NP_ING: {
		LexiconType.VERB: [ArgumentType.PP, ArgumentType.ING],
		LexiconType.NOUN: [ArgumentType.PP, ArgumentType.ING]
	},
	SubcatType.NOM_NP_P_NP_ING: {
		LexiconType.VERB: [ArgumentType.OBJ, ArgumentType.PP, ArgumentType.ING],
		LexiconType.NOUN: [ArgumentType.OBJ, ArgumentType.PP, ArgumentType.ING]
	},
	SubcatType.NOM_POSSING: {
		LexiconType.VERB: [ArgumentType.ING],
		LexiconType.NOUN: [ArgumentType.ING]
	},
	SubcatType.NOM_P_POSSING: {
		LexiconType.VERB: [ArgumentType.ING],
		LexiconType.NOUN: [ArgumentType.ING]
	},
	SubcatType.NOM_PP_P_POSSING: {
		LexiconType.VERB: [ArgumentType.PP, ArgumentType.ING],
		LexiconType.NOUN: [ArgumentType.PP, ArgumentType.ING]
	},
	SubcatType.NOM_POSSING_PP: {
		LexiconType.VERB: [ArgumentType.ING, ArgumentType.PP],
		LexiconType.NOUN: [ArgumentType.ING, ArgumentType.PP]
	},
	SubcatType.NOM_NP_P_POSSING: {
		LexiconType.VERB: [ArgumentType.OBJ, ArgumentType.ING],
		LexiconType.NOUN: [ArgumentType.OBJ, ArgumentType.ING]
	},
	SubcatType.NOM_FOR_TO_INF: {
		LexiconType.VERB: [ArgumentType.TO_INF],
		LexiconType.NOUN: [ArgumentType.TO_INF]
	},
	SubcatType.NOM_NP_TO_INF_OC: {
		LexiconType.VERB: [ArgumentType.OBJ, ArgumentType.TO_INF],
		LexiconType.NOUN: [ArgumentType.OBJ, ArgumentType.TO_INF]
	},
	SubcatType.NOM_NP_TO_INF_SC: {
		LexiconType.VERB: [ArgumentType.OBJ, ArgumentType.TO_INF],
		LexiconType.NOUN: [ArgumentType.OBJ, ArgumentType.TO_INF]
	},
	SubcatType.NOM_NP_TO_INF_VC: {
		LexiconType.VERB: [ArgumentType.OBJ, ArgumentType.TO_INF],
		LexiconType.NOUN: [ArgumentType.OBJ, ArgumentType.TO_INF]
	},
	SubcatType.NOM_TO_INF_SC: {
		LexiconType.VERB: [ArgumentType.TO_INF],
		LexiconType.NOUN: [ArgumentType.TO_INF]
	},
	SubcatType.NOM_P_NP_TO_INF_OC: {
		LexiconType.VERB: [ArgumentType.PP, ArgumentType.TO_INF],
		LexiconType.NOUN: [ArgumentType.PP, ArgumentType.TO_INF]
	},
	SubcatType.NOM_P_NP_TO_INF: {
		LexiconType.VERB: [ArgumentType.PP, ArgumentType.TO_INF],
		LexiconType.NOUN: [ArgumentType.PP, ArgumentType.TO_INF]
	},
	SubcatType.NOM_P_NP_TO_INF_VC: {
		LexiconType.VERB: [ArgumentType.PP, ArgumentType.TO_INF],
		LexiconType.NOUN: [ArgumentType.PP, ArgumentType.TO_INF]
	},
	SubcatType.NOM_PP_FOR_TO_INF: {
		LexiconType.VERB: [ArgumentType.PP, ArgumentType.TO_INF],
		LexiconType.NOUN: [ArgumentType.PP, ArgumentType.TO_INF]
	},
	SubcatType.NOM_S: {
		LexiconType.VERB: [ArgumentType.SBAR],
		LexiconType.NOUN: [ArgumentType.SBAR]
	},
	SubcatType.NOM_THAT_S: {
		LexiconType.VERB: [ArgumentType.SBAR],
		LexiconType.NOUN: [ArgumentType.SBAR]
	},
	SubcatType.NOM_S_SUBJUNCT: {
		LexiconType.VERB: [ArgumentType.SBAR],
		LexiconType.NOUN: [ArgumentType.SBAR]
	},
	SubcatType.NOM_NP_S: {
		LexiconType.VERB: [ArgumentType.OBJ, ArgumentType.SBAR],
		LexiconType.NOUN: [ArgumentType.OBJ, ArgumentType.SBAR]
	},
	SubcatType.NOM_PP_THAT_S: {
		LexiconType.VERB: [ArgumentType.PP, ArgumentType.SBAR],
		LexiconType.NOUN: [ArgumentType.PP, ArgumentType.SBAR]
	},
	SubcatType.NOM_PP_THAT_S_SUBJUNCT: {
		LexiconType.VERB: [ArgumentType.PP, ArgumentType.SBAR],
		LexiconType.NOUN: [ArgumentType.PP, ArgumentType.SBAR]
	},
	SubcatType.NOM_NP_AS_IF_S_SUBJUNCT: {
		LexiconType.VERB: [ArgumentType.OBJ, ArgumentType.SBAR],
		LexiconType.NOUN: [ArgumentType.OBJ, ArgumentType.SBAR]
	},
	SubcatType.NOM_WH_S: {
		LexiconType.VERB: [ArgumentType.SBAR],
		LexiconType.NOUN: [ArgumentType.SBAR]
	},
	SubcatType.NOM_WHERE_WHEN_S: {
		LexiconType.VERB: [ArgumentType.SBAR],
		LexiconType.NOUN: [ArgumentType.SBAR]
	},
	SubcatType.NOM_HOW_S: {
		LexiconType.VERB: [ArgumentType.SBAR],
		LexiconType.NOUN: [ArgumentType.SBAR]
	},
	SubcatType.NOM_PP_HOW_TO_INF: {
		LexiconType.VERB: [ArgumentType.SBAR],
		LexiconType.NOUN: [ArgumentType.SBAR]
	},
	SubcatType.NOM_NP_WH_S: {
		LexiconType.VERB: [ArgumentType.OBJ, ArgumentType.SBAR],
		LexiconType.NOUN: [ArgumentType.SBAR]
	},
	SubcatType.NOM_P_WH_S: {
		LexiconType.VERB: [ArgumentType.SBAR],
		LexiconType.NOUN: [ArgumentType.SBAR]
	},
	SubcatType.NOM_PP_WH_S: {
		LexiconType.VERB: [ArgumentType.PP, ArgumentType.SBAR],
		LexiconType.NOUN: [ArgumentType.PP, ArgumentType.SBAR]
	},
	SubcatType.NOM_PP_P_WH_S: {
		LexiconType.VERB: [ArgumentType.PP, ArgumentType.SBAR],
		LexiconType.NOUN: [ArgumentType.PP, ArgumentType.SBAR]
	},
	SubcatType.NOM_NP_P_WH_S: {
		LexiconType.VERB: [ArgumentType.OBJ, ArgumentType.SBAR],
		LexiconType.NOUN: [ArgumentType.SBAR]
	}
}

SUBCAT_OPTIONAL_ARGS = {
	SubcatType.NOM_PP_HOW_TO_INF: {
		LexiconType.VERB: [ArgumentType.PP],
		LexiconType.NOUN: [ArgumentType.PP]
	},
}


def get_default_requires(subcat_type: SubcatType, lexicon_type: LexiconType) -> List[ArgumentType]:
	if subcat_type in SUBCAT_REQUIRED_ARGS:
		return SUBCAT_REQUIRED_ARGS[subcat_type][lexicon_type]

	return []


def get_default_optionals(subcat_type: SubcatType, lexicon_type: LexiconType) -> List[ArgumentType]:
	if subcat_type in SUBCAT_OPTIONAL_ARGS:
		return SUBCAT_OPTIONAL_ARGS[subcat_type][lexicon_type]

	return []
