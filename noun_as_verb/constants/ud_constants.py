########################################################################################################################
# Universal POS tags

UPOS_VERB = "VERB"
UPOS_PART = "PART"
UPOS_ADP = "ADP"
UPOS_ADJ = "ADJ"
UPOS_ADV = "ADV"
UPOS_NOUN = "NOUN"
UPOS_PROPN = "PROPN"
UPOS_PRON = "PRON"
UPOS_DET = "DET"
UPOS_AUX = "AUX"
UPOS_PUNCT = "PUNCT"



########################################################################################################################
# NOT Universal pos TAGS

TAG_NNS = "NNS"
TAG_NNPS = "NNPS"
TAG_TO = "TO"



########################################################################################################################
# Universal dependency relations

# General universal relations
URELATION_NMOD = "nmod"
URELATION_NSUBJ = "nsubj"
URELATION_NSUBJPASS = "nsubjpass"
URELATION_DOBJ = "dobj"
URELATION_IOBJ = "iobj"
URELATION_ADVMOD = "advmod"
URELATION_AMOD = "amod"
URELATION_ADVCL = "advcl"
URELATION_XCOMP = "xcomp"
URELATION_CCOMP = "ccomp"
URELATION_ACL = "acl"
URELATION_ACL_RELCL = "acl:relcl"
URELATION_NMOD_POSS = "nmod:poss"
URELATION_COMPOUND = "compound"
URELATION_COP = "cop"
URELATION_PRT = "prt"
URELATION_COMPOUND_PRT = "compound:prt"

# Specific relations
URELATION_TO = "mark_" + TAG_TO		# For "to" preceeded TO-INF (must be right before a verb)
URELATION_THAT = "mark_that"		# For "that" preceeded S
URELATION_ANY = "any"				# wild card ud relation