from arguments_extractor.constants.lexicon_constants import *
from arguments_extractor.constants.ud_constants import *

########################################################################################################################
# Translation of UD tags to positions
# {UD-LINK: (VERB-POS, NOM-POS)}
LINK_TO_POS = {
	URELATION_NMOD: 		([POS_PREFIX, POS_TO_INF],						[POS_PREFIX, POS_TO_INF]),		# TO-INF gets NMOD on "to-be" cluases
	URELATION_NSUBJ: 		([POS_NSUBJ],									[POS_NSUBJ]),
	URELATION_NSUBJPASS: 	([POS_NSUBJPASS],								[]),
	URELATION_DOBJ:			([POS_DOBJ],									[]),
	URELATION_IOBJ:			([POS_IOBJ],									[]),
	URELATION_ADVMOD:		([POS_ADMOD, POS_PREFIX],						[POS_ADMOD, POS_PREFIX]),		# advmod can be PREFIX for prepositions of 2 words, like "next to" and "out of"
	URELATION_AMOD:			([],											[POS_AJMOD]),
	URELATION_ADVCL:		([POS_PREFIX, POS_ING, POS_TO_INF],				[]),
	URELATION_XCOMP:		([POS_PREFIX, POS_ING, POS_TO_INF],				[]),
	URELATION_CCOMP:		([POS_PREFIX, POS_ING, POS_THAT, POS_SBAR],		[POS_THAT]),
	URELATION_ACL:			([POS_ACL],										[POS_PREFIX, POS_ING, POS_TO_INF]),
	URELATION_ACL_RELCL:	([],											[POS_THAT]),
	URELATION_NMOD_POSS:	([POS_DET_POSS],								[POS_DET_POSS]),
	URELATION_COMPOUND:		([],											[POS_N_N_MOD]),
	URELATION_PRT:			([POS_PART],									[]),
	URELATION_COMPOUND_PRT:	([POS_PART],									[]),
}