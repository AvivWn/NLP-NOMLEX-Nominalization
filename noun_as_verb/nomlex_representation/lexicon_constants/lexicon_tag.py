from enum import Enum


class LexiconTag(str, Enum):
	# This enum consider all the deprecated tags from NOMLEX original lexicons
	# The tags are used to parse the lexicon, but does not exist in the new one

	# entry features
	SUBJ_OBJ_ALT = "SUBJ-OBJ-ALT"
	SUBJ_IND_OBJ_ALT = "SUBJ-IND-OBJ-ALT"

	# subcat properties
	ALTERNATES_OPT = "ALTERNATES-OPT"

	# argument types
	ADVAL = "ADVAL"
	ADVAL_NOM = "ADVAL-NOM"
	PVAL = "PVAL"
	PVAL1 = "PVAL1"
	PVAL2 = "PVAL2"
	PVAL_NOM = "PVAL-NOM"

	# argument properties
	DET_POSS_ONLY = "DET-POSS-ONLY"
	N_N_MOD_ONLY = "N-N-MOD-ONLY"

	# argument values
	NOT_PP_BY = "NOT-PP-BY"
	P_LOC = "p-loc"
	P_DIR = "p-dir"

	# noun tags
	PLUR_ONLY = "PLUR-ONLY"
	SING_ONLY = "SING-ONLY"
	EXISTS = "EXISTS"
	RARE_NOUN = "RARE-NOUN"
	RARE_NOM = "RARE-NOM"
