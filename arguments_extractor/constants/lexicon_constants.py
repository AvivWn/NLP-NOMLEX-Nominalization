# This files contains all the constants that appear in the lexicon

########################################################################################################################
# Types of arguments\complements
COMP_SUBJ = "SUBJECT"
COMP_OBJ = "OBJECT"
COMP_IND_OBJ = "IND-OBJ"
COMP_P_IND_OBJ = "P-IND-OBJ"
COMP_PP = "PP"
COMP_PP1 = "PP1"
COMP_PP2 = "PP2"
COMP_FOR_NP = "FOR-NP"
COMP_ADVP = "ADVP"
COMP_ADJP = "ADJP"
COMP_AS_NP_OC = "AS-NP-OC"
COMP_AS_NP_SC = "AS-NP-SC"

COMP_NP = "NP"
COMP_ING_NPC = "ING-NPC"
COMP_ING_OC = "ING-OC"
COMP_ING_SC = "ING-SC"
COMP_ING_POSSC = "ING-POSSC"
COMP_AS_ING_OC = "AS-ING-OC"
COMP_P_ING_NPC = "P-ING-NPC"
COMP_P_ING_OC = "P-ING-OC"
COMP_P_ING_SC = "P-ING-SC"
COMP_P_ING_POSSC = "P-ING-POSSC"
COMP_P_NP_ING = "P-NP-ING"
COMP_POSS_ING_SC = "POSS-ING-SC"
COMP_POSS_ING_VC = "POSS-ING-VC"

COMP_TO_INF = "TO-INF"
COMP_TO_INF_OC = "TO-INF-OC"
COMP_TO_INF_SC = "TO-INF-SC"
COMP_TO_INF_VC = "TO-INF-VC"
COMP_TO_INF_P_OC = "TO-INF-P-OC"
#COMP_FOR_NP_TO_INF = "FOR-NP-TO-INF"
COMP_TO_INF_FOR_OC = "TO-INF-FOR-OC"

COMP_SBAR = "SBAR"
#COMP_SBAR_SUBJUNCT = "SBAR-SUBJUNCT"
#COMP_AS_IF_S_SUBJUNCT = "AS-IF-S-SUBJUNCT"
COMP_AS_IF_S = "AS-IF-S"
COMP_WH_S = "WH-S"
COMP_WHERE_WHEN_S = "WHERE-WHEN-S"
COMP_HOW_S = "HOW-S"
COMP_HOW_TO_INF = "HOW-TO-INF"
COMP_P_WH_S = "P-WH-S"
#COMP_P_SECOND_SUBJECT = "COMP-P-SECOND-SUBJECT"

COMP_PART = "PARTICLE"
COMP_INSTRUMENT = "INSTRUMENT"
#COMP_INSTRUMENT = "INSTRUMENT"

# Arguments\Complements that appear only in the original representation of the lexicon
OLD_COMP_PVAL = "PVAL"
OLD_COMP_PVAL1 = "PVAL1"
OLD_COMP_PVAL2 = "PVAL2"
OLD_COMP_PVAL_NOM = "PVAL-NOM"
OLD_COMP_PVAL_NOM1 = "PVAL-NOM1"
OLD_COMP_PVAL_NOM2 = "PVAL-NOM2"
OLD_COMP_ADVAL = "ADVAL"
OLD_COMP_ADVAL_NOM = "ADVAL-NOM"
OLD_COMP_P_OBJ = "P-OBJ"

# Complements that are used temporarily and are not a legitable COMP in the new representation
IGNORE_COMP = "IGNORE-COMPLEMENT" # = The original complement should be ignored and deleted




########################################################################################################################
# Type of entries

ENT_ORTH = "ORTH"
ENT_VERB_SUBC = "VERB-SUBC"
ENT_SINGULAR_FALSE = "SINGULAR-FALSE"

ENT_NOM_TYPE = "NOM-TYPE"
TYPE_OF_NOM = "TYPE-OF-NOM"
NOM_TYPE_VERB_NOM = "VERB-NOM"
NOM_TYPE_SUBJ = "SUBJECT"
NOM_TYPE_OBJ = "OBJECT"
NOM_TYPE_IND_OBJ = "IND-OBJ"
NOM_TYPE_INSTRUMENT = "INSTRUMENT"
NOM_TYPE_P_OBJ = "P-OBJ"
TYPE_PART = "PARTICLE"
TYPE_PP = "PP"


# Entries that link to other entries
ENT_NEXT = "NEXT"
ENT_NOM = "NOM"
ENT_VERB = "VERB"
ENT_SINGULAR = "SINGULAR"
ENT_PLURAL = "PLURAL"

# Removed from the new representation
ENT_VERB_SUBJ = "VERB-SUBJ"

ENT_FEATURES = "FEATURES"
FEATURE_SUBJ_OBJ_ALT = "SUBJ-OBJ-ALT"
FEATURE_SUBJ_IND_OBJ_ALT = "SUBJ-IND-OBJ-ALT"

# Entries that are not in used
ENT_NOUN = "NOUN"
ENT_NOUN_SUBC = "NOUN-SUBC"
ENT_SUBJ_ATTRIBUTE = "SUBJ-ATTRIBUTE"
ENT_OBJ_ATTRIBUTE = "OBJ-ATTRIBUTE"
ENT_IND_OBJ_ATTRIBUTE = "IND-OBJ-ATTRIBUTE"
ENT_SEMI_AUTOMATIC = "SEMI-AUTOMATIC"
ENT_PLURAL_FREQ = "PLURAL-FREQ"
#ENT_SUBJECT_PP_OF_FREQ = "SUBJECT-PP-OF-FREQ"



########################################################################################################################
# Type of possible positions for arugments

POS_N_N_MOD = "N-N-MOD"
POS_DET_POSS = "DET-POSS"
POS_NSUBJ = "NSUBJ"
POS_DOBJ = "DOBJ"
POS_IOBJ = "IOBJ"
POS_AJMOD = "AJMOD"
POS_ADMOD = "ADMOD"
POS_TO_INF = "TO-INF"
POS_SBAR = "SBAR"
POS_THAT = "THAT"
POS_ING = "ING"
POS_ACL = "ACL"

POS_PREFIX = "PREFIX" # other positions, like arguments that starts with prepositions
POS_NOM = "NOM"

# Positions that are used temporarily and are not a legitable POS in the new representation
OPT_POS = "OPTIONAL-POSITION" # = The position of a complement is optional



########################################################################################################################
# Constants list of values

# Constants preposition types, and their appropriate values
P_DIR = "p-dir"
P_DIR_REPLACEMENTS = ["down", "across", "from", "into", "off", "onto",
					  "out", "out of", "outside", "over", "past", "through",
					  "to", "toward", "towards", "up", "up to", "via"]
P_LOC = "p-loc"
P_LOC_REPLACEMENTS = ["above", "about", "against", "along", "around", "behind", "below", "beneath",
					  "beside", "between", "beyond", "by", "in", "inside", "near", "next to", "on",
					  "throughout", "under", "upon", "within"]

WH_VERB_OPTIONS = ["if", "whether", "what"]
WH_NOM_OPTIONS = ["whether", "what"]
WHERE_WHEN_OPTIONS = ["where", "when", "how many", "how much"]
HOW_OPTIONS = ["how"]
HOW_TO_OPTIONS = ["how to"]
POSSESIVE_OPTIONS = ["my", "your", "his", "our", "her", "their"]

NONE_VALUES = ["NONE", "*NONE*", "none"] # Means that there is no possible value for that entry



########################################################################################################################
# Argument possitions, properties and constraints


# Possible positions for the arguments, splitted into two different lists
ARG_CONSTANTS = "CONSTANTS"
ARG_PREFIXES = "PREFIXES"

# Basic argument properties
ARG_ILLEGAL_PREFIXES = "ILLEGAL-PREFIXES"


# Root constraints

# Required patterns for the root of the argument
ARG_ROOT_PATTERNS = "ROOT-PATTERNS"
PATTERN_ING = "ing$"

# Required u-postags and dependency u-relations for the the root of the argument (u=universal)
ARG_ROOT_UPOSTAGS = "ROOT-UPOSTAGS"
ARG_ROOT_RELATIONS = "ROOT-RELATIONS"


# Argument boolean constraints
ARG_CONSTRAINTS = "CONSTRAINTS"
ARG_CONSTRAINT_POSSESSIVE = "POSSESSIVE"
ARG_CONSTRAINT_N_N_MOD_NO_OTHER_OBJ = "N-N-MOD-NO-OTHER-OBJ"
ARG_CONSTRAINT_DET_POSS_NO_OTHER_OBJ = "DET-POSS-NO-OTHER-OBJ"
ARG_CONSTRAINT_PLURAL = "PLURAL"
ARG_CONSTRAINT_SUBJUNCT = "SUBJUNCT"

# Argument properties that depends on other arguments
ARG_INCLUDING = "INCLUDING" # Relevant for plural when subject should be plural
ARG_CONTIGUOUS_TO = "CONTIGUOUS" # the arguemnt that is right before that argument (the two arguments must be adjacent)
ARG_CONTROLLED = "CONTROLLED"

# The argument can be linked to the nom, the verb or ANY other possible complement
LINKED_NOM = "NOM"
LINKED_VERB = "VERB"



########################################################################################################################
# Subcategorization constraints

SUBCAT_NOT = "NOT"
SUBCAT_REQUIRED = "REQUIRED"
SUBCAT_OPTIONAL = "OPTIONAL"

SUBCAT_CONSTRAINTS = "CONSTRAINTS"
SUBCAT_CONSTRAINT_ADVP_OR_ADJP = "ADVP_OR_ADJP"
SUBCAT_CONSTRAINT_ALTERNATES = "ALTERNATES"
OLD_SUBCAT_CONSTRAINT_ALTERNATES_OPT = "ALTERNATES-OPT" # only in the original representation