from typing import List, Dict, Union

from yet_another_verb.nomlex.constants import LexiconType, SubcatType, ArgumentType, ArgumentValue
from yet_another_verb.nomlex.constants.lexicon_type import LEXICON_TYPES

ArgValues = List[ArgumentValue]
ArgValuesByLexicon = Dict[LexiconType, ArgValues]
ArgValuesByArgType = Dict[ArgumentType, ArgValues]
ArgTypeByLexicon = Dict[LexiconType, ArgumentType]
ArgValuesByArgAndLexicon = Dict[LexiconType, ArgValuesByArgType]


def _get_values_by_lexicon(
		arg_type: Union[ArgumentType, ArgTypeByLexicon],
		arg_values: Union[ArgValues, ArgValuesByLexicon],
) -> ArgValuesByArgAndLexicon:
	def choose_values(lexicon_type: LexiconType) -> ArgValuesByArgType:
		used_values = arg_values.get(lexicon_type, []) if isinstance(arg_values, dict) else arg_values
		used_type = arg_type.get(lexicon_type, []) if isinstance(arg_type, dict) else arg_type
		return {used_type: used_values}

	return {lexicon_type: choose_values(lexicon_type) for lexicon_type in LEXICON_TYPES}


def _get_modifier_values(adjp_only: bool = False) -> ArgValuesByArgAndLexicon:
	return _get_values_by_lexicon(ArgumentType.MODIFIER, arg_values={
		LexiconType.VERB: [ArgumentValue.AJMOD] if adjp_only else [ArgumentValue.ADMOD],
		LexiconType.NOUN: [ArgumentValue.AJMOD] if adjp_only else [ArgumentValue.ADMOD, ArgumentValue.AJMOD]
	})


def _get_ing_values(optional_poss: Union[bool, Dict[LexiconType, bool]] = False) -> ArgValuesByArgAndLexicon:
	def _get_values(is_optional_poss: bool):
		return [ArgumentValue.ING, ArgumentValue.POSSING] if is_optional_poss else [ArgumentValue.ING]

	if isinstance(optional_poss, bool):
		arg_values = _get_values(optional_poss)
	else:
		arg_values = {t: _get_values(optional_poss.get(t, False)) for t in LEXICON_TYPES}

	return _get_values_by_lexicon(ArgumentType.ING, arg_values)


def _get_to_inf_values() -> ArgValuesByArgAndLexicon:
	return _get_values_by_lexicon(ArgumentType.TO_INF, [ArgumentValue.TO_INF])


def _get_sbar_values(required_that: bool = False) -> ArgValuesByArgAndLexicon:
	return _get_values_by_lexicon(ArgumentType.SBAR, arg_values={
		LexiconType.VERB: [ArgumentValue.THAT_S] if required_that else [ArgumentValue.SBAR, ArgumentValue.THAT_S],
		LexiconType.NOUN: [ArgumentValue.THAT_S]
	})


def _get_wh_s_values() -> ArgValuesByArgAndLexicon:
	return _get_values_by_lexicon(ArgumentType.SBAR, arg_values={
		LexiconType.VERB: [ArgumentValue.WHAT_S, ArgumentValue.WHETHER_S, ArgumentValue.IF_S],
		LexiconType.NOUN: [ArgumentValue.WHAT_S, ArgumentValue.WHETHER_S]
	})


def _get_where_when_s_values() -> ArgValuesByArgAndLexicon:
	return _get_values_by_lexicon(ArgumentType.SBAR, arg_values=[
		ArgumentValue.WHEN_S, ArgumentValue.WHERE_S, ArgumentValue.HOW_MANY_S, ArgumentValue.HOW_MUCH_S
	])


def _get_how_s_values(untensed_only: bool = False) -> ArgValuesByArgAndLexicon:
	return _get_values_by_lexicon(ArgumentType.SBAR, arg_values={
		LexiconType.VERB: [ArgumentValue.HOW_TO_INF] if untensed_only else [ArgumentValue.HOW_S],
		LexiconType.NOUN: [ArgumentValue.HOW_TO_INF] if untensed_only else [ArgumentValue.HOW_S]
	})


def _get_as_if_s_values() -> ArgValuesByArgAndLexicon:
	return _get_values_by_lexicon(ArgumentType.SBAR, arg_values=[ArgumentValue.AS_IF_S])


OPTIONAL_POSS_VERB = {LexiconType.VERB: True, LexiconType.NOUN: False}

ARG_DEFAULT_VALUES = {
	SubcatType.NOM_ADVP: _get_modifier_values(),
	SubcatType.NOM_NP_ADVP: _get_modifier_values(),
	SubcatType.NOM_ADVP_PP: _get_modifier_values(),
	SubcatType.NOM_NP_AS_ADJP: _get_modifier_values(adjp_only=True),

	SubcatType.NOM_NP_AS_ING: _get_ing_values(),
	SubcatType.NOM_ING_SC: _get_ing_values(optional_poss=OPTIONAL_POSS_VERB),
	SubcatType.NOM_PART_ING_SC: _get_ing_values(optional_poss=OPTIONAL_POSS_VERB),
	SubcatType.NOM_NP_ING: _get_ing_values(),
	SubcatType.NOM_NP_ING_OC: _get_ing_values(),
	SubcatType.NOM_NP_ING_SC: _get_ing_values(),
	SubcatType.NOM_P_ING_SC: _get_ing_values(optional_poss=OPTIONAL_POSS_VERB),
	SubcatType.NOM_NP_P_ING: _get_ing_values(),
	SubcatType.NOM_NP_P_ING_OC: _get_ing_values(),
	SubcatType.NOM_NP_P_ING_SC: _get_ing_values(),
	SubcatType.NOM_P_NP_ING: _get_ing_values(),
	SubcatType.NOM_NP_P_NP_ING: _get_ing_values(),
	SubcatType.NOM_POSSING: _get_ing_values(optional_poss=True),
	SubcatType.NOM_P_POSSING: _get_ing_values(optional_poss=True),
	SubcatType.NOM_PP_P_POSSING: _get_ing_values(optional_poss=True),
	SubcatType.NOM_POSSING_PP: _get_ing_values(optional_poss=True),
	SubcatType.NOM_NP_P_POSSING: _get_ing_values(optional_poss=True),

	SubcatType.NOM_FOR_TO_INF: _get_to_inf_values(),
	SubcatType.NOM_NP_TO_INF_OC: _get_to_inf_values(),
	SubcatType.NOM_NP_TO_INF_SC: _get_to_inf_values(),
	SubcatType.NOM_NP_TO_INF_VC: _get_to_inf_values(),
	SubcatType.NOM_TO_INF_SC: _get_to_inf_values(),
	SubcatType.NOM_P_NP_TO_INF_OC: _get_to_inf_values(),
	SubcatType.NOM_P_NP_TO_INF: _get_to_inf_values(),  # why is it different from NOM-P-NP-TO-INF-OC ?????,
	SubcatType.NOM_P_NP_TO_INF_VC: _get_to_inf_values(),
	SubcatType.NOM_PP_FOR_TO_INF: _get_to_inf_values(),

	SubcatType.NOM_S: _get_sbar_values(),
	SubcatType.NOM_THAT_S: _get_sbar_values(required_that=True),
	SubcatType.NOM_S_SUBJUNCT: _get_sbar_values(),  # Means that the verb of SBAR is in subjunctive mood
	SubcatType.NOM_NP_S: _get_sbar_values(),
	SubcatType.NOM_PP_THAT_S: _get_sbar_values(required_that=True),
	SubcatType.NOM_PP_THAT_S_SUBJUNCT: _get_sbar_values(required_that=True),
	SubcatType.NOM_NP_AS_IF_S_SUBJUNCT: _get_as_if_s_values(),
	SubcatType.NOM_WH_S: _get_wh_s_values(),
	SubcatType.NOM_WHERE_WHEN_S: _get_where_when_s_values(),
	SubcatType.NOM_HOW_S: _get_how_s_values(),
	SubcatType.NOM_PP_HOW_TO_INF: _get_how_s_values(untensed_only=True),
	SubcatType.NOM_NP_WH_S: _get_wh_s_values(),
	SubcatType.NOM_P_WH_S: _get_wh_s_values(),
	SubcatType.NOM_PP_WH_S: _get_wh_s_values(),
	SubcatType.NOM_PP_P_WH_S: _get_wh_s_values(),
	SubcatType.NOM_NP_P_WH_S: _get_wh_s_values(),
}


def get_default_arg_values(subcat_type: SubcatType, arg_type: ArgumentType, lexicon_type: LexiconType) -> ArgValues:
	arg_values = ARG_DEFAULT_VALUES.get(subcat_type, {}).get(lexicon_type, {}).get(arg_type, [])

	if len(arg_values) == 0:
		if ArgumentType.is_pp_arg(arg_type) or ArgumentType.is_np_arg(arg_type):
			arg_values.append(ArgumentValue.NP)
		elif arg_type == ArgumentType.PART:
			arg_values.append(ArgumentValue.PART)
		else:
			raise NotImplementedError()

	return arg_values
