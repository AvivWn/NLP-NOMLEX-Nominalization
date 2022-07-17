from typing import List, Dict, Union
from itertools import chain

from yet_another_verb.nomlex.constants import LexiconType, SubcatType
from yet_another_verb.arguments_extractor.extraction import ArgumentType

Preps = List[str]
PrepsByLexicon = Dict[LexiconType, Preps]
PrepsByArgType = Dict[ArgumentType, Preps]
ArgTypeByLexicon = Dict[LexiconType, ArgumentType]
PrepsByArgAndLexicon = Dict[LexiconType, PrepsByArgType]


def _get_preps_by_lexicon(
		arg_type: Union[ArgumentType, ArgTypeByLexicon],
		preps: Union[Preps, PrepsByLexicon] = None
) -> PrepsByArgAndLexicon:
	all_lexicon_types = [LexiconType.VERB, LexiconType.NOUN]
	preps = preps if preps is not None else []

	def choose_preps(lexicon_type: LexiconType) -> PrepsByArgType:
		used_preps = preps.get(lexicon_type, []) if isinstance(preps, dict) else preps
		used_type = arg_type.get(lexicon_type, []) if isinstance(arg_type, dict) else arg_type
		return {used_type: used_preps}

	return {lexicon_type: choose_preps(lexicon_type) for lexicon_type in all_lexicon_types}


def _combine_preps(multiple_defaults: List[PrepsByArgAndLexicon]) -> PrepsByArgAndLexicon:
	return {key: dict(chain.from_iterable(d[key].items() for d in multiple_defaults)) for key in LexiconType}


NOUN_OF_PREP = {
	LexiconType.NOUN: ["of"]
}

ARG_DEFAULTS_PREPS = {
	SubcatType.NOM_NP_AS_ADJP: _get_preps_by_lexicon(ArgumentType.MODIFIER, ["as"]),
	SubcatType.NOM_NP_AS_ING: _get_preps_by_lexicon(ArgumentType.ING, ["as"]),
	SubcatType.NOM_AS_NP: _get_preps_by_lexicon(ArgumentType.PP, ["as"]),
	SubcatType.NOM_PART_AS_NP: _get_preps_by_lexicon(ArgumentType.PP, ["as"]),
	SubcatType.NOM_NP_AS_NP: _get_preps_by_lexicon(ArgumentType.PP, ["as"]),
	SubcatType.NOM_PART_NP_AS_NP: _get_preps_by_lexicon(ArgumentType.PP, ["as"]),
	SubcatType.NOM_NP_AS_NP_SC: _get_preps_by_lexicon(ArgumentType.PP, ["as"]),
	SubcatType.NOM_NP_PP_AS_NP: _get_preps_by_lexicon(ArgumentType.PP, ["as"]),
	SubcatType.NOM_ING_SC: _get_preps_by_lexicon(ArgumentType.ING, NOUN_OF_PREP),
	SubcatType.NOM_PART_ING_SC: _get_preps_by_lexicon(ArgumentType.ING, NOUN_OF_PREP),
	SubcatType.NOM_POSSING: _get_preps_by_lexicon(ArgumentType.ING, NOUN_OF_PREP),
	SubcatType.NOM_PP_P_POSSING: _combine_preps([
		_get_preps_by_lexicon(ArgumentType.ING, ["about", "on"]),
		_get_preps_by_lexicon(ArgumentType.PP, ["between", "among", "with"])
	]),
	SubcatType.NOM_POSSING_PP: _get_preps_by_lexicon(ArgumentType.ING, NOUN_OF_PREP),
	SubcatType.NOM_FOR_TO_INF: _get_preps_by_lexicon(ArgumentType.TO_INF, ["for"]),
	SubcatType.NOM_PP_FOR_TO_INF: _get_preps_by_lexicon(ArgumentType.TO_INF, ["for"]),
	SubcatType.NOM_WH_S: _get_preps_by_lexicon(ArgumentType.SBAR, NOUN_OF_PREP),
	SubcatType.NOM_WHERE_WHEN_S: _get_preps_by_lexicon(ArgumentType.SBAR, NOUN_OF_PREP),
	SubcatType.NOM_HOW_S: _get_preps_by_lexicon(ArgumentType.SBAR, NOUN_OF_PREP),
	SubcatType.NOM_PP_HOW_TO_INF: _get_preps_by_lexicon(ArgumentType.SBAR, NOUN_OF_PREP),
	SubcatType.NOM_NP_WH_S: _get_preps_by_lexicon(ArgumentType.SBAR, NOUN_OF_PREP),
	SubcatType.NOM_PP_WH_S: _get_preps_by_lexicon(ArgumentType.SBAR, NOUN_OF_PREP)
}


def get_default_preps_by_arg(subcat_type: SubcatType, lexicon_type: LexiconType) -> PrepsByArgType:
	return ARG_DEFAULTS_PREPS.get(subcat_type, {}).get(lexicon_type, {})
