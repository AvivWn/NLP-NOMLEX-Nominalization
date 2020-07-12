from arguments_extractor.constants.lexicon_constants import *

########################################################################################################################
# Translation of UD tags to positions
# {UD-LINK: (VERB-POS, NOM-POS)}
LINK_TO_POS = {
	"nmod": 		([POS_PREFIX, POS_TO_INF],						[POS_PREFIX, POS_TO_INF]),		# TO-INF gets NMOD on "to-be" cluases
	"nsubj": 		([POS_NSUBJ],									[POS_NSUBJ]),
	"nsubjpass": 	([POS_NSUBJPASS],								[]),
	"dobj":			([POS_DOBJ],									[]),				# UD V1
	"obj":			([POS_DOBJ],									[]),				# UD V2
	"iobj":			([POS_IOBJ],									[]),
	"advmod":		([POS_ADMOD],									[POS_ADMOD]),
	"amod":			([],											[POS_AJMOD]),
	"advcl":		([POS_PREFIX, POS_ING, POS_TO_INF],				[]),
	"xcomp":		([POS_PREFIX, POS_ING, POS_TO_INF],				[]),
	"ccomp":		([POS_PREFIX, POS_ING, POS_THAT, POS_SBAR],		[POS_THAT]),
	"acl":			([POS_ACL],										[POS_PREFIX, POS_ING, POS_TO_INF]),
	"acl:relcl":	([],											[POS_THAT]),
	"nmod:poss":	([POS_DET_POSS],								[POS_DET_POSS]),
	"compound":		([],											[POS_N_N_MOD]),

	"prt":			([POS_PREFIX],									[]),
	"compound:prt":	([POS_PREFIX],									[]),
}