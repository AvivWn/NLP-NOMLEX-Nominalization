from typing import Dict

from yet_another_verb.nomlex.constants import LexiconType, SubcatType
from yet_another_verb.arguments_extractor.extraction import ArgumentType

ARG_RENAMINGS = {
	SubcatType.NOM_PP_PP: {
		LexiconType.VERB: {ArgumentType.PP: ArgumentType.PP1},
		LexiconType.NOUN: {ArgumentType.PP: ArgumentType.PP1}
	},
	SubcatType.NOM_NP_PP_PP: {
		LexiconType.VERB: {ArgumentType.PP: ArgumentType.PP1},
		LexiconType.NOUN: {ArgumentType.PP: ArgumentType.PP1}
	},
	SubcatType.NOM_NP_ING: {
		LexiconType.VERB: {ArgumentType.OBJ: ArgumentType.NP},
		LexiconType.NOUN: {ArgumentType.OBJ: ArgumentType.NP}
	},
	SubcatType.NOM_P_ING_SC: {
		LexiconType.VERB: {ArgumentType.PP: ArgumentType.ING},
		LexiconType.NOUN: {ArgumentType.PP: ArgumentType.ING}
	},
	SubcatType.NOM_NP_P_ING: {
		LexiconType.VERB: {ArgumentType.PP: ArgumentType.ING, ArgumentType.OBJ: ArgumentType.NP},
		LexiconType.NOUN: {ArgumentType.PP: ArgumentType.ING, ArgumentType.OBJ: ArgumentType.NP}
	},
	SubcatType.NOM_NP_P_ING_OC: {
		LexiconType.VERB: {ArgumentType.PP: ArgumentType.ING},
		LexiconType.NOUN: {ArgumentType.PP: ArgumentType.ING}
	},
	SubcatType.NOM_NP_P_ING_SC: {
		LexiconType.VERB: {ArgumentType.PP: ArgumentType.ING},
		LexiconType.NOUN: {ArgumentType.PP: ArgumentType.ING}
	},
	SubcatType.NOM_P_POSSING: {
		LexiconType.VERB: {ArgumentType.PP: ArgumentType.ING},
		LexiconType.NOUN: {ArgumentType.PP: ArgumentType.ING}
	},
	SubcatType.NOM_PP_P_POSSING: {
		LexiconType.VERB: {ArgumentType.PP: ArgumentType.ING, ArgumentType.PP1: ArgumentType.PP},
		LexiconType.NOUN: {ArgumentType.PP: ArgumentType.ING, ArgumentType.PP1: ArgumentType.PP}
	},
	SubcatType.NOM_NP_P_POSSING: {
		LexiconType.VERB: {ArgumentType.PP: ArgumentType.ING},
		LexiconType.NOUN: {ArgumentType.PP: ArgumentType.ING}
	},
	SubcatType.NOM_PP_FOR_TO_INF: {
		LexiconType.VERB: {ArgumentType.PP1: ArgumentType.PP},
		LexiconType.NOUN: {ArgumentType.PP1: ArgumentType.PP}
	},
	SubcatType.NOM_PP_THAT_S: {
		LexiconType.VERB: {ArgumentType.PP1: ArgumentType.PP},
		LexiconType.NOUN: {ArgumentType.PP1: ArgumentType.PP}
	},
	SubcatType.NOM_PP_THAT_S_SUBJUNCT: {
		LexiconType.VERB: {ArgumentType.PP1: ArgumentType.PP},
		LexiconType.NOUN: {ArgumentType.PP1: ArgumentType.PP}
	},
	SubcatType.NOM_PP_HOW_TO_INF: {
		LexiconType.VERB: {ArgumentType.PP1: ArgumentType.PP},
		LexiconType.NOUN: {ArgumentType.PP1: ArgumentType.PP}
	},
	SubcatType.NOM_P_WH_S: {
		LexiconType.VERB: {ArgumentType.PP: ArgumentType.SBAR},
		LexiconType.NOUN: {ArgumentType.PP: ArgumentType.SBAR}
	},
	SubcatType.NOM_PP_WH_S: {
		LexiconType.VERB: {ArgumentType.PP1: ArgumentType.PP},
		LexiconType.NOUN: {ArgumentType.PP1: ArgumentType.PP}
	},
	SubcatType.NOM_PP_P_WH_S: {
		LexiconType.VERB: {ArgumentType.PP: ArgumentType.SBAR, ArgumentType.PP1: ArgumentType.PP},
		LexiconType.NOUN: {ArgumentType.PP: ArgumentType.SBAR, ArgumentType.PP1: ArgumentType.PP}
	},
	SubcatType.NOM_NP_P_WH_S: {
		LexiconType.VERB: {ArgumentType.PP: ArgumentType.SBAR},
		LexiconType.NOUN: {ArgumentType.PP: ArgumentType.SBAR}
	}
}


def get_argument_renamings(subcat_type: SubcatType, lexicon_type: LexiconType) -> Dict[ArgumentType, ArgumentType]:
	if subcat_type in ARG_RENAMINGS:
		return ARG_RENAMINGS[subcat_type][lexicon_type]

	return {}
