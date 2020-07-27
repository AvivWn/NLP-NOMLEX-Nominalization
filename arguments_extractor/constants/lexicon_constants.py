# This files contains all the constants that appear in the lexicon

########################################################################################################################
# Types of arguments\complements

# Noun Phrases Complements
COMP_SUBJ = "SUBJECT"
COMP_SECOND_SUBJ = "SECOND-SUBJECT"
COMP_OBJ = "OBJECT"
COMP_IND_OBJ = "IND-OBJ"
COMP_NP = "OBJECT"

# Prepositional Complements
COMP_PP = "PP"
COMP_PP1 = "PP1"
COMP_PP2 = "PP2"
# COMP_P_NP = "P-NP"				# A preposition phrase that must contains a noun
COMP_AS_NP_OC = "AS-NP-OC"		# Object Controlled
COMP_AS_NP_SC = "AS-NP-SC"		# Subject Controlled

# Modifier Complements
COMP_ADVP = "ADVP"
COMP_ADJP = "ADJP"
COMP_AS_ADJP = "AS-ADJP"

# ING Complements
COMP_ING_NPC = "ING-OC"		# NP Controlled
COMP_ING_OC = "ING-OC"		# Object Controlled
COMP_ING_SC = "ING-SC"		# Subject Controlled
COMP_ING_POC = "ING-POC"	# Preposition Object Controlled
COMP_ING_ARBC = "ING-ARBC"	# Arbitrary Controlled
COMP_POSSING = "POSSING"	# Possessive Gerund

# P-ING Complements
COMP_AS_ING_OC = "AS-ING-OC"
COMP_P_ING_NPC = "P-ING-OC"
COMP_P_ING_OC = "P-ING-OC"
COMP_P_ING_SC = "P-ING-SC"
COMP_P_ING_ARBC = "P-ING-ARBC"
COMP_P_POSSING = "P-POSSING"

# TO-INF Complements
COMP_TO_INF = "TO-INF"
COMP_TO_INF_OC = "TO-INF-OC"
COMP_TO_INF_SC = "TO-INF-SC"
COMP_TO_INF_VC = "TO-INF-VC"
COMP_TO_INF_POC = "TO-INF-POC"
COMP_FOR_TO_INF = "FOR-TO-INF"

# SBAR Complements
COMP_SBAR = "SBAR"
COMP_WH_S = "WH-S"
COMP_WHERE_WHEN_S = "WHERE-WHEN-S"
COMP_HOW_S = "HOW-S"
COMP_HOW_TO_INF = "HOW-TO-INF"
COMP_P_WH_S = "P-WH-S"

# SBAR-SUBJUNCT Complements
# Currently they are just standard SBAR complements (still not sure how to detect the SUBJUNCTIVE MOOD)
COMP_SBAR_SUBJUNCT = "SBAR"
COMP_AS_IF_S_SUBJUNCT = "AS-IF-S"

COMP_PART = "PARTICLE"
# COMP_INSTRUMENT = "INSTRUMENT"



# Arguments\Complements that appear only in the original representation of the lexicon
OLD_COMP_PVAL = "PVAL"
OLD_COMP_PVAL1 = "PVAL1"
OLD_COMP_PVAL2 = "PVAL2"
OLD_COMP_PVAL_NOM = "PVAL-NOM"
OLD_COMP_PVAL_NOM1 = "PVAL-NOM1"
OLD_COMP_PVAL_NOM2 = "PVAL-NOM2"
OLD_COMP_ADVAL = "ADVAL"
OLD_COMP_ADVAL_NOM = "ADVAL-NOM"

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
# NOM_TYPE_INSTRUMENT = "INSTRUMENT"
# NOM_TYPE_P_OBJ = "P-OBJ"
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

DEFAULT_ENTRY = "DEFAULT"



########################################################################################################################
# Type of possible positions for arugments

POS_N_N_MOD = "N-N-MOD"
POS_DET_POSS = "DET-POSS"
POS_NSUBJ = "NSUBJ"
POS_NSUBJPASS = "NSUBJPASS"
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
ADD_POS = "ADDITIONAL-POSITION" # = The position should be added to the existing ones



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
POSSESIVE_OPTIONS = ["my", "your", "his", "our", "her", "their", "its"]

NONE_VALUES = ["NONE", "*NONE*", "none"] # Means that there is no possible value for that entry



########################################################################################################################
# Argument possitions, properties and constraints


# Possible positions for the arguments, splitted into two different lists
ARG_POSITIONS = "POSITIONS"

# Basic argument properties
ARG_PREFIXES = "PREFIXES"
ARG_ILLEGAL_PREFIXES = "ILLEGAL-PREFIXES"


# Root constraints

# Required patterns for the root of the argument
ARG_ROOT_PATTERNS = "ROOT-PATTERNS"
PATTERN_ING = "ing$"

# Required u-postags and dependency u-relations for the the root of the argument (u=universal)
ARG_ROOT_UPOSTAGS = "ROOT-UPOSTAGS"
ARG_ROOT_URELATIONS = "ROOT-URELATIONS"


# Argument boolean constraints
ARG_CONSTRAINTS = "CONSTRAINTS"
ARG_CONSTRAINT_REQUIRED_PREFIX = "REQUIRED-PREFIX" # Prefix is required for all positions
ARG_CONSTRAINT_OPTIONAL_POSSESSIVE = "OPTIONAL-POSSESSIVE"
ARG_CONSTRAINT_N_N_MOD_NO_OTHER_OBJ = "N-N-MOD-NO-OTHER-OBJ"
ARG_CONSTRAINT_DET_POSS_NO_OTHER_OBJ = "DET-POSS-NO-OTHER-OBJ"
ARG_CONSTRAINT_PLURAL = "PLURAL"
ARG_CONSTRAINT_SUBJUNCT = "SUBJUNCT"

# Argument properties that depends on other arguments
ARG_INCLUDING = "INCLUDING" 		# An argument may include another argument (relevant for plurality constraints)
ARG_CONTIGUOUS_TO = "CONTIGUOUS" 	# the arguemnt that is right before that argument (the two arguments must be adjacent)
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
SUBCAT_CONSTRAINT_ADVP_OR_ADJP = "ADVP-OR-ADJP"
SUBCAT_CONSTRAINT_ALTERNATES = "ALTERNATES"
OLD_SUBCAT_CONSTRAINT_ALTERNATES_OPT = "ALTERNATES-OPT" # only in the original representation

DEFAULT_SUBCAT = "DEFAULT"