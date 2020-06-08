from lexicon_constants import *

# This file includes several "tables" (dictionary) for needed modifications for the lexicon, based on the manual



# Dictionary that includes fixes for specific subcats, based on the manual
# The subcats in this dictionary are assumed to be "the known subcats"
# {SUBCAT: ((VERB_REQUIRED, VERB_DEFAULTS, VERB_TRANSLATION, VERB_SPECIAL_VALUES),
# 		    (NOM_REQUIRED, NOM_OPTIONAL, NOM_TRANSLATIONS, NOM_SPECIAL_VALUES))}
lexicon_fixes_dict = {
	"NOM-INTRANS": 				(([], 												{},																{},																	{}),
								 ([], 												{},																{},																	{})),
	"NOM-INTRANS-RECIP": 		(([], 												{},																{},																	{}),
								 ([], 												{}, 															{},																	{})),


	# NP and PP
	"NOM-NP": 					(([COMP_OBJ], 										{}, 															{},																	{}),
								 ([], 												{}, 															{},																	{})),
	"NOM-NP-NP": 				(([COMP_IND_OBJ, COMP_OBJ],							{COMP_IND_OBJ: [POS_IOBJ]}, 									{COMP_IND_OBJ: IGNORE_COMP},										{}),
								 ([COMP_IND_OBJ], 									{},																{},																	{})),
	"NOM-PP": 					(([COMP_PP], 										{}, 															{},																	{}),
								 ([COMP_PP], 										{}, 															{},																	{})),
	"NOM-PP-PP": 				(([COMP_PP1, COMP_PP2], 							{}, 															{COMP_PP: COMP_PP1},												{}),
								 ([COMP_PP1, COMP_PP2], 							{},																{COMP_PP: COMP_PP1},												{})),
	"NOM-NP-PP": 				(([COMP_OBJ, COMP_PP], 								{}, 															{},																	{}),
								 ([COMP_PP],										{},																{},																	{})),
	"NOM-NP-TO-NP": 			(([COMP_OBJ, COMP_IND_OBJ],							{COMP_IND_OBJ: ["to", POS_IOBJ]},								{COMP_IND_OBJ: IGNORE_COMP},										{}),
								 ([COMP_IND_OBJ],									{COMP_IND_OBJ: ["to"]},											{},																	{})),
	"NOM-NP-FOR-NP": 			(([COMP_OBJ, COMP_IND_OBJ],							{COMP_IND_OBJ: ["for", POS_IOBJ]},								{COMP_IND_OBJ: IGNORE_COMP},										{}),
								 ([COMP_IND_OBJ],									{COMP_IND_OBJ: ["for"]},										{},																	{})),
	"NOM-NP-PP-PP": 			(([COMP_OBJ, COMP_PP1, COMP_PP2],					{},																{COMP_PP: COMP_PP1},												{}),
								 ([COMP_PP1, COMP_PP2],								{},																{COMP_PP: COMP_PP1},												{})),


	# ADVP
	"NOM-ADVP": 				(([COMP_ADVP], 										{COMP_ADVP: [POS_ADMOD]}, 										{},																	{}),
								 ([], 												{COMP_ADVP:[POS_ADMOD,OPT_POS], COMP_ADJP:[POS_AJMOD,OPT_POS]},	{},																	{})),
	"NOM-NP-ADVP": 				(([COMP_OBJ, COMP_ADVP], 							{COMP_ADVP: [POS_ADMOD]}, 										{},																	{}),
								 ([],												{COMP_ADVP:[POS_ADMOD,OPT_POS], COMP_ADJP:[POS_AJMOD,OPT_POS]}, {},																	{})),
	"NOM-ADVP-PP": 				(([COMP_ADVP, COMP_PP], 							{COMP_ADVP: [POS_ADMOD]}, 										{},																	{}),
								 ([COMP_PP],										{COMP_ADVP:[POS_ADMOD,OPT_POS], COMP_ADJP:[POS_AJMOD,OPT_POS]},	{},																	{})),


	# AS PHRASE
	"NOM-NP-AS-ING":	 		(([COMP_OBJ, COMP_AS_ING_OC], 						{COMP_AS_ING_OC: ["as"]}, 										{},																	{}),
								 ([COMP_OBJ, COMP_AS_ING_OC], 						{COMP_AS_ING_OC: ["as"]},										{},																	{})),
	"NOM-NP-AS-ADJP": 			(([COMP_OBJ, COMP_ADJP], 							{COMP_ADJP: ["as"]}, 											{},																	{}),
								 ([COMP_OBJ, COMP_ADJP], 							{COMP_ADJP: ["as"]},											{},																	{})),
	"NOM-AS-NP": 				(([COMP_SUBJ, COMP_AS_NP_SC], 						{COMP_AS_NP_SC: ["as"]}, 										{},																	{}),
								 ([COMP_SUBJ, COMP_AS_NP_SC], 						{COMP_AS_NP_SC: ["as"]},										{},																	{})),
	"NOM-NP-AS-NP": 			(([COMP_OBJ, COMP_AS_NP_OC], 						{COMP_AS_NP_OC: ["as"]}, 										{},																	{}),
								 ([COMP_OBJ, COMP_AS_NP_OC],						{COMP_AS_NP_OC: ["as"]},										{},																	{})),
	"NOM-NP-AS-NP-SC": 			(([COMP_SUBJ, COMP_OBJ, COMP_AS_NP_SC], 			{COMP_AS_NP_SC: ["as"]}, 										{},																	{}),
								 ([COMP_SUBJ, COMP_AS_NP_SC],						{COMP_AS_NP_SC: ["as"]},										{},																	{})),
	"NOM-NP-PP-AS-NP": 			(([COMP_OBJ, COMP_P_IND_OBJ, COMP_AS_NP_OC], 		{COMP_AS_NP_OC: ["as"]}, 										{COMP_PP1: COMP_P_IND_OBJ},											{}),
								 ([COMP_OBJ, COMP_P_IND_OBJ, COMP_AS_NP_OC],		{COMP_AS_NP_OC: ["as"]},										{COMP_PP1: COMP_P_IND_OBJ},											{})),


	# GERUNDS
	"NOM-NP-ING": 				(([COMP_NP, COMP_ING_NPC],							{}, 															{COMP_OBJ: IGNORE_COMP},											{}),
								 ([COMP_NP, COMP_ING_NPC],							{},																{COMP_OBJ: COMP_ING_NPC},											{})),
	"NOM-NP-ING-OC": 			(([COMP_OBJ, COMP_ING_OC], 							{},																{COMP_OBJ: IGNORE_COMP},											{}),
								 ([COMP_OBJ, COMP_ING_OC],							{},																{},																	{})),
	"NOM-NP-ING-SC": 			(([COMP_OBJ, COMP_ING_SC], 							{}, 															{COMP_OBJ: IGNORE_COMP},											{}),
								 ([COMP_OBJ, COMP_ING_SC],							{},																{},																	{})),
	"NOM-NP-P-ING": 			(([COMP_NP, COMP_P_ING_NPC],						{},																{COMP_PP: COMP_P_ING_NPC, COMP_OBJ: IGNORE_COMP},					{COMP_P_ING_NPC: ["NOM-SUBC", "P-", "PVAL"]}),
								 ([COMP_NP, COMP_P_ING_NPC], 						{}, 															{COMP_PP: COMP_P_ING_NPC, COMP_OBJ: COMP_NP}, 						{COMP_P_ING_NPC: ["NOM-SUBC", "P-", "PVAL"]})), # NOM-SUBC for both
	"NOM-NP-P-ING-OC": 			(([COMP_OBJ, COMP_P_ING_OC], 						{}, 															{COMP_PP: COMP_P_ING_OC, COMP_OBJ: IGNORE_COMP}, 					{COMP_P_ING_OC: ["NOM-SUBC", "P-", "PVAL"]}),
								 ([COMP_OBJ, COMP_P_ING_OC], 						{}, 															{COMP_PP: COMP_P_ING_OC}, 											{COMP_P_ING_OC: ["NOM-SUBC", "P-", "PVAL"]})), # NOM-SUBC for both
	"NOM-NP-P-ING-SC": 			(([COMP_OBJ, COMP_P_ING_SC],						{}, 															{COMP_PP: COMP_P_ING_SC, COMP_OBJ: IGNORE_COMP}, 					{COMP_P_ING_SC: ["NOM-SUBC", "P-", "PVAL"]}),
								 ([COMP_OBJ, COMP_P_ING_SC],	 					{}, 															{COMP_PP: COMP_P_ING_SC}, 											{COMP_P_ING_SC: ["NOM-SUBC", "P-", "PVAL"]})), # NOM-SUBC for both


	# POSSESIVE GERUNDS
	"NOM-ING-SC": 				(([COMP_ING_SC], 									{COMP_POSS_ING_SC: [OPT_POS]}, 									{},																	{}),
								 ([COMP_ING_SC],									{COMP_ING_SC: ["of"]},											{},																	{COMP_ING_SC: ["NOM-SUBC", "P-", "PVAL"]})), # NOM-SUBC just for nom
	"NOM-POSSING": 				(([COMP_ING_POSSC], 								{COMP_POSS_ING_VC: [OPT_POS]}, 									{},																	{}),
								 ([COMP_ING_POSSC],									{COMP_POSS_ING_VC: [OPT_POS], COMP_ING_POSSC: ["of"]},			{},																	{COMP_ING_POSSC: ["NOM-SUBC", "P-", "PVAL"]})), # NOM-SUBC just for nom
	"NOM-P-ING-SC": 			(([COMP_P_ING_SC],								 	{COMP_POSS_ING_SC: [OPT_POS]}, 									{COMP_PP: COMP_P_ING_SC}, 											{COMP_P_ING_SC: ["NOM-SUBC", "P-", "PVAL"]}),
								 ([COMP_P_ING_SC],									{}, 															{COMP_PP: COMP_P_ING_SC}, 											{COMP_P_ING_SC: ["NOM-SUBC", "P-", "PVAL"]})), # NOM-SUBC for both
	"NOM-P-POSSING": 			(([COMP_P_ING_POSSC], 								{COMP_POSS_ING_VC: [OPT_POS]},									{COMP_PP: COMP_P_ING_POSSC},										{COMP_P_ING_POSSC: ["NOM-SUBC", "P-", "PVAL"]}),
								 ([COMP_P_ING_POSSC], 								{COMP_POSS_ING_VC: [OPT_POS]}, 									{COMP_PP: COMP_P_ING_POSSC}, 										{COMP_P_ING_POSSC: ["NOM-SUBC", "P-", "PVAL"]})), # NOM-SUBC for both
	"NOM-PP-P-POSSING": 		(([COMP_P_IND_OBJ, COMP_P_ING_POSSC], 				{COMP_POSS_ING_VC: [OPT_POS]}, 									{COMP_PP1: COMP_P_IND_OBJ, COMP_PP: COMP_P_ING_POSSC}, 				{}),
								 ([COMP_P_IND_OBJ, COMP_P_ING_POSSC],				{COMP_POSS_ING_VC: [OPT_POS], COMP_P_ING_POSSC: ["with"]},		{COMP_PP1: COMP_P_IND_OBJ, COMP_PP: COMP_P_ING_POSSC},				{})),
	"NOM-P-NP-ING": 			(([COMP_P_NP_ING], 									{}, 															{COMP_PP: COMP_P_NP_ING}, 											{}),
								 ([COMP_P_NP_ING], 									{}, 															{COMP_PP: COMP_P_NP_ING}, 											{})),
	"NOM-POSSING-PP": 			(([COMP_ING_POSSC, COMP_PP], 						{COMP_POSS_ING_VC: [OPT_POS]}, 									{},																	{}),
								 ([COMP_ING_POSSC, COMP_PP],						{COMP_POSS_ING_VC: [OPT_POS], COMP_ING_POSSC: ["of"]},			{},																	{COMP_ING_POSSC: ["NOM-SUBC", "P-", "PVAL"]})), # NOM-SUBC just for nom
	"NOM-NP-P-POSSING": 		(([COMP_OBJ, COMP_P_ING_POSSC],						{COMP_POSS_ING_VC: [OPT_POS]},									{COMP_PP: COMP_P_ING_POSSC},										{COMP_P_ING_POSSC: ["NOM-SUBC", "P-", "PVAL"]}),
								 ([COMP_OBJ, COMP_P_ING_POSSC], 					{COMP_POSS_ING_VC: [OPT_POS]}, 									{COMP_PP: COMP_P_ING_POSSC}, 										{COMP_P_ING_POSSC: ["NOM-SUBC", "P-", "PVAL"]})), # NOM-SUBC for both
	"NOM-NP-P-NP-ING": 			(([COMP_OBJ, COMP_P_NP_ING], 						{}, 															{COMP_PP: COMP_P_NP_ING}, 											{}),
								 ([COMP_OBJ, COMP_P_NP_ING], 						{}, 															{COMP_PP: COMP_P_NP_ING}, 											{})),


	# INFINITIVE
	"NOM-FOR-TO-INF": 			(([COMP_FOR_NP, COMP_TO_INF_FOR_OC],				{COMP_FOR_NP: ["for"], COMP_TO_INF_FOR_OC: [POS_TO_INF]}, 		{},																	{}),
								 ([COMP_FOR_NP, COMP_TO_INF_FOR_OC],				{COMP_FOR_NP: ["for"], COMP_TO_INF_FOR_OC: [POS_TO_INF]},		{},																	{})),
	#"NOM-NP-TO-INF":			(([COMP_OBJ, FOR_TO_INF_OR_TO_INF], 				{COMP_TO_INF: [POS_TO_INF], COMP_FOR_NP_TO_INF: ["for"]},		{},																	{}),
	#					 		 ([COMP_OBJ, FOR_TO_INF_OR_TO_INF], 				{COMP_TO_INF: [POS_TO_INF], COMP_FOR_NP_TO_INF: ["for"]},		{}, 																{})), # {COMP_TO_INF: ["NOM-SUBC", "FOR-TO-INF"]}), # Not appear in the manual
	"NOM-NP-TO-INF-OC": 		(([COMP_OBJ, COMP_TO_INF_OC], 						{COMP_TO_INF_OC: [POS_TO_INF]}, 								{},																	{}),
								 ([COMP_OBJ, COMP_TO_INF_OC],						{COMP_TO_INF_OC: [POS_TO_INF]},									{},																	{})),
	"NOM-NP-TO-INF-SC": 		(([COMP_OBJ, COMP_TO_INF_SC], 						{COMP_TO_INF_SC: [POS_TO_INF]}, 								{},																	{}),
								 ([COMP_OBJ, COMP_TO_INF_SC],						{COMP_TO_INF_SC: [POS_TO_INF]},									{},																	{})),
	"NOM-NP-TO-INF-VC": 		(([COMP_OBJ, COMP_TO_INF_VC], 						{COMP_TO_INF_VC: [POS_TO_INF]}, 								{},																	{}),
								 ([COMP_OBJ, COMP_TO_INF_VC],						{COMP_TO_INF_VC: [POS_TO_INF]},									{},																	{})),
	"NOM-TO-INF-SC": 			(([COMP_TO_INF_SC], 								{COMP_TO_INF_SC: [POS_TO_INF]}, 								{},																	{}),
								 ([COMP_TO_INF_SC],									{COMP_TO_INF_SC: [POS_TO_INF]},									{},																	{})),
	"NOM-P-NP-TO-INF-OC":		(([COMP_PP, COMP_TO_INF_P_OC], 						{COMP_TO_INF_P_OC: [POS_TO_INF]}, 								{},																	{}),
								 ([COMP_PP, COMP_TO_INF_P_OC],						{COMP_TO_INF_P_OC: [POS_TO_INF]},								{},																	{})),
	"NOM-P-NP-TO-INF":			(([COMP_PP, COMP_TO_INF_P_OC], 						{COMP_TO_INF_P_OC: [POS_TO_INF]}, 								{},																	{}),
								 ([COMP_PP, COMP_TO_INF_P_OC],						{COMP_TO_INF_P_OC: [POS_TO_INF]},								{},																	{})), # why is it different from NOM-P-NP-TO-INF-OC ?????
	"NOM-P-NP-TO-INF-VC":		(([COMP_PP, COMP_TO_INF_VC], 						{COMP_TO_INF_VC: [POS_TO_INF]}, 								{},																	{}),
								 ([COMP_PP, COMP_TO_INF_VC],						{COMP_TO_INF_VC: [POS_TO_INF]},									{},																	{})),
	"NOM-PP-FOR-TO-INF":		(([COMP_P_IND_OBJ,COMP_FOR_NP,COMP_TO_INF_FOR_OC],	{COMP_FOR_NP: ["for"], COMP_TO_INF_FOR_OC: [POS_TO_INF]},		{COMP_PP1: COMP_P_IND_OBJ},											{}),
								 ([COMP_P_IND_OBJ,COMP_FOR_NP,COMP_TO_INF_FOR_OC],	{COMP_FOR_NP: ["for"], COMP_TO_INF_FOR_OC: [POS_TO_INF]},		{COMP_PP1: COMP_P_IND_OBJ},											{})),
	"NOM-PP-TO-INF-RECIP":		(([COMP_TO_INF],	 								{COMP_P_IND_OBJ: ["with", OPT_POS]},							{COMP_PP1: COMP_P_IND_OBJ},											{}),
								 ([COMP_TO_INF, COMP_P_IND_OBJ], 					{COMP_P_IND_OBJ: ["with"]},										{COMP_PP1: COMP_P_IND_OBJ},											{})), # SUBJECT = SUBJECT and PVAL NP


	# SBAR
	"NOM-S":					(([COMP_SBAR],										{COMP_SBAR: [POS_SBAR, POS_THAT]}, 								{},																	{}),
								 ([COMP_SBAR], 										{COMP_SBAR: [POS_THAT]},										{},																	{})),
	"NOM-THAT-S":				(([COMP_SBAR],										{COMP_SBAR: [POS_THAT]}, 										{},																	{}),
								 ([COMP_SBAR], 										{COMP_SBAR: [POS_THAT]},										{},																	{})),
	"NOM-S-SUBJUNCT":			(([COMP_SBAR],										{COMP_SBAR: [POS_SBAR, POS_THAT]},								{},																	{}),
								 ([COMP_SBAR], 										{COMP_SBAR: [POS_THAT]},										{},																	{})), # Means that the verb of SBAR is subjunctive
	"NOM-NP-S":					(([COMP_OBJ, COMP_SBAR],							{COMP_SBAR: [POS_SBAR, POS_THAT]},								{},																	{}),
								 ([COMP_OBJ, COMP_SBAR], 							{COMP_SBAR: [POS_THAT]},										{},																	{})),
	"NOM-PP-THAT-S":			(([COMP_P_IND_OBJ, COMP_SBAR],						{COMP_SBAR: [POS_THAT]},										{COMP_PP1: COMP_P_IND_OBJ},											{}),
								 ([COMP_P_IND_OBJ, COMP_SBAR], 						{COMP_SBAR: [POS_THAT]},										{COMP_PP1: COMP_P_IND_OBJ},											{})),
	"NOM-PP-THAT-S-SUBJUNCT":	(([COMP_P_IND_OBJ, COMP_SBAR],						{COMP_SBAR: [POS_THAT]},										{COMP_PP1: COMP_P_IND_OBJ},											{}),
								 ([COMP_P_IND_OBJ, COMP_SBAR], 						{COMP_SBAR: [POS_THAT]},										{COMP_PP1: COMP_P_IND_OBJ},											{})),
	"NOM-NP-AS-IF-S-SUBJUNCT":	(([COMP_OBJ, COMP_AS_IF_S],							{COMP_AS_IF_S: ["as if"]},										{},																	{}),
								 ([COMP_OBJ, COMP_AS_IF_S], 						{COMP_AS_IF_S: ["as if"]},										{},																	{})),


	# WH-SBAR
	# There must be a WH-word after the preposition that preceds the WH-phase. The WH-word is determined based on the complement's name:
	# - WH-S, P-WH-S: if (only for verb), whether, what
	# - WHERE-WHERN-S: where, when, how-quant-n (how many and how much)
	# - HOW-S: how
	# - HOW-TO-INF: how to
	# If where-when tag appears under P-WH in NOM-SUBC, than there are probably more possible tags for both nom and verb [?????]. This occurs only for NOM-WH-S.
	"NOM-WH-S":					(([COMP_WH_S],									{COMP_WH_S: WH_VERB_OPTIONS},									{},																	{}),
								 ([COMP_WH_S], 									{COMP_WH_S: ["of"]},											{},																	{COMP_WH_S: ["NOM-SUBC", "P-WH", "PVAL"]})), # NOM-SUBC for just nom
	"NOM-WHERE-WHEN-S":			(([COMP_WHERE_WHEN_S],							{COMP_WHERE_WHEN_S: WHERE_WHEN_OPTIONS},						{},																	{}),
								 ([COMP_WHERE_WHEN_S], 							{COMP_WHERE_WHEN_S: ["of"]},									{},																	{COMP_WHERE_WHEN_S: ["NOM-SUBC", "P-WH", "PVAL"]})), # NOM-SUBC for just nom
	"NOM-HOW-S":				(([COMP_HOW_S],									{COMP_HOW_S: HOW_OPTIONS},										{},																	{}),
								 ([COMP_HOW_S], 								{COMP_HOW_S: ["of"]},											{},																	{COMP_HOW_S: ["NOM-SUBC", "P-WH", "PVAL"]})), # NOM-SUBC for just nom
	"NOM-PP-HOW-TO-INF":		(([COMP_HOW_TO_INF],							{COMP_HOW_TO_INF: HOW_TO_OPTIONS, COMP_P_IND_OBJ: [OPT_POS]},	{COMP_PP1: COMP_P_IND_OBJ},											{}),
								 ([COMP_HOW_TO_INF],							{COMP_HOW_TO_INF: ["of"], COMP_P_IND_OBJ: [OPT_POS]},			{COMP_PP1: COMP_P_IND_OBJ},											{COMP_HOW_TO_INF: ["NOM-SUBC", "P-WH", "PVAL"]})), # NOM-SUBC for just nom
	"NOM-NP-WH-S":				(([COMP_OBJ, COMP_WH_S],						{COMP_WH_S: WH_VERB_OPTIONS},									{},																	{}),
								 ([COMP_WH_S], 									{COMP_WH_S: ["of"]},											{},																	{COMP_WH_S: ["NOM-SUBC", "P-WH", "PVAL"]})), # NOM-SUBC for just nom
	"NOM-P-WH-S":				(([COMP_P_WH_S], 								{}, 															{COMP_PP: COMP_P_WH_S}, 											{COMP_P_WH_S: ["NOM-SUBC", "P-WH", "PVAL"]}),
								 ([COMP_P_WH_S], 								{},						 										{COMP_PP: COMP_P_WH_S}, 											{COMP_P_WH_S: ["NOM-SUBC", "P-WH", "PVAL"]})), # NOM-SUBC for both
	"NOM-PP-WH-S":				(([COMP_P_IND_OBJ, COMP_WH_S],					{COMP_WH_S: WH_VERB_OPTIONS},									{COMP_PP1: COMP_P_IND_OBJ},											{}),
								 ([COMP_P_IND_OBJ, COMP_WH_S],					{COMP_WH_S: ["of"]},											{COMP_PP1: COMP_P_IND_OBJ},											{COMP_WH_S: ["NOM-SUBC", "P-WH", "PVAL"]})), # NOM-SUBC for just nom
	"NOM-PP-P-WH-S":			(([COMP_P_IND_OBJ, COMP_P_WH_S], 				{}, 															{COMP_PP1: COMP_P_IND_OBJ, COMP_PP: COMP_P_WH_S}, 					{COMP_P_WH_S: ["NOM-SUBC", "P-WH", "PVAL"]}),
								 ([COMP_P_IND_OBJ, COMP_P_WH_S], 				{},	 															{COMP_PP1: COMP_P_IND_OBJ, COMP_PP: COMP_P_WH_S}, 					{COMP_P_WH_S: ["NOM-SUBC", "P-WH", "PVAL"]})), # NOM-SUBC for both
	"NOM-NP-P-WH-S":			(([COMP_OBJ, COMP_P_WH_S], 						{}, 															{COMP_PP: COMP_P_WH_S}, 											{COMP_P_WH_S: ["NOM-SUBC", "P-WH", "PVAL"]}),
								 ([COMP_P_WH_S], 								{}, 															{COMP_PP: COMP_P_WH_S}, 											{COMP_P_WH_S: ["NOM-SUBC", "P-WH", "PVAL"]})), # NOM-SUBC for both

}

# Dicationary that includes a more complex argument positions structure, similarly to the default column in the above dictionary
# {SUBCAT: (VERB_CONSTRAINTS, NOM_CONSTRAINTS)}
complex_argument_positions = {
	"NOM-NP-ING":				({COMP_NP: [{COMP_ING_NPC: [POS_NSUBJ, POS_POSS]}], COMP_ING_NPC: [POS_ING]},
								 {COMP_ING_NPC: [POS_ING]}), # Maybe COMP_NP need more positions for nom
	"NOM-NP-ING-OC":			({COMP_OBJ: [POS_DOBJ], COMP_ING_OC: [POS_ING, {COMP_OBJ: [POS_ACL]}]},
								 {COMP_ING_OC: [POS_ING]}),
	"NOM-NP-ING-SC":			({COMP_OBJ: [POS_DOBJ], COMP_ING_SC: [POS_ING]},
								 {COMP_ING_SC: [POS_ING]}),

	"NOM-ING-SC":				({COMP_ING_SC: [POS_ING], COMP_POSS_ING_SC: [{COMP_ING_SC: [POS_NSUBJ, POS_POSS]}]},
								 {COMP_ING_SC: [POS_ING]}),
	"NOM-POSSING":				({COMP_ING_POSSC: [POS_ING], COMP_POSS_ING_VC: [{COMP_ING_POSSC: [POS_NSUBJ, POS_POSS]}]},
								 {COMP_ING_POSSC: [POS_ING], COMP_POSS_ING_VC: [{COMP_ING_POSSC: [POS_NSUBJ, POS_POSS]}]}),
	"NOM-P-ING-SC":				({COMP_POSS_ING_SC: [{COMP_P_ING_SC: [POS_NSUBJ, POS_POSS]}]},
								 {}),
	"NOM-P-POSSING":			({COMP_POSS_ING_VC: [{COMP_P_ING_POSSC: [POS_NSUBJ, POS_POSS]}]},
								 {COMP_POSS_ING_VC: [{COMP_P_ING_POSSC: [POS_NSUBJ, POS_POSS]}]}),
	"NOM-PP-P-POSSING":			({COMP_POSS_ING_VC: [{COMP_P_ING_POSSC: [POS_NSUBJ, POS_POSS]}]},
								 {COMP_POSS_ING_VC: [{COMP_P_ING_POSSC: [POS_NSUBJ, POS_POSS]}]}),
	"NOM-POSSING-PP":			({COMP_ING_POSSC: [POS_ING], COMP_POSS_ING_VC: [{COMP_ING_POSSC: [POS_NSUBJ, POS_POSS]}]},
								 {COMP_ING_POSSC: [POS_ING], COMP_POSS_ING_VC: [{COMP_ING_POSSC: [POS_NSUBJ, POS_POSS]}]}),
	"NOM-NP-P-POSSING": 		({COMP_POSS_ING_VC: [{COMP_P_ING_POSSC: [POS_NSUBJ, POS_POSS]}]},
								 {COMP_POSS_ING_VC: [{COMP_P_ING_POSSC: [POS_NSUBJ, POS_POSS]}]})
}

# Dicationary that includes a list of constraints for specific subcats
# {SUBCAT: (VERB_CONSTRAINTS, NOM_CONSTRAINTS)}
subcat_constraints = {
	"NOM-ADVP":					([],
								 [SUBCAT_CONSTRAINT_ADVP_OR_ADJP]),
	"NOM-NP-ADVP":				([],
								 [SUBCAT_CONSTRAINT_ADVP_OR_ADJP]),
	"NOM-ADVP-PP":				([],
								 [SUBCAT_CONSTRAINT_ADVP_OR_ADJP])
}

# Dicationary that includes a list of constraints for arguments of specific subcats
# {SUBCAT: (VERB_CONSTRAINTS, NOM_CONSTRAINTS)}
argument_constraints = {
	# PLURAL and INCLUDING
	"NOM-INTRANS-RECIP": 		({COMP_SUBJ: {ARG_PLURAL: True}},
								 {COMP_SUBJ: {ARG_PLURAL: True}}),
	"NOM-PP-TO-INF-RECIP":		({COMP_SUBJ: {ARG_HEAD_POSTAGS: [POSTAG_VB], ARG_INCLUDING: COMP_P_IND_OBJ, ARG_PLURAL: True}},
								 {COMP_SUBJ: {ARG_HEAD_POSTAGS: [POSTAG_VB], ARG_INCLUDING: COMP_P_IND_OBJ, ARG_PLURAL: True}}),

	"NOM-NP-NP":				({},
								 {COMP_OBJ: {ARG_CONTIGUOUS_TO: COMP_IND_OBJ}}), # ONLY FOR VERB

	# AS NP
	"NOM-AS-NP":				({COMP_AS_NP_SC: {ARG_CONTROLLED: [COMP_SUBJ], ARG_ILLEGAL_PREFIXES: ["as if"]}},
								 {COMP_AS_NP_SC: {ARG_CONTROLLED: [COMP_SUBJ], ARG_ILLEGAL_PREFIXES: ["as if"]}}),
	"NOM-NP-AS-NP":				({COMP_AS_NP_OC: {ARG_CONTROLLED: [COMP_OBJ], ARG_ILLEGAL_PREFIXES: ["as if"]}},
								 {COMP_AS_NP_OC: {ARG_CONTROLLED: [COMP_OBJ], ARG_ILLEGAL_PREFIXES: ["as if"]}}),
	"NOM-NP-AS-NP-SC":			({COMP_AS_NP_SC: {ARG_CONTROLLED: [COMP_SUBJ], ARG_ILLEGAL_PREFIXES: ["as if"]}},
								 {COMP_AS_NP_SC: {ARG_CONTROLLED: [COMP_SUBJ], ARG_ILLEGAL_PREFIXES: ["as if"]}}),
	"NOM-NP-PP-AS-NP":			({COMP_AS_NP_OC: {ARG_CONTROLLED: [COMP_OBJ], ARG_ILLEGAL_PREFIXES: ["as if"]}},
								 {COMP_AS_NP_OC: {ARG_CONTROLLED: [COMP_OBJ], ARG_ILLEGAL_PREFIXES: ["as if"]}}),

	# ING
	"NOM-NP-AS-ING":			({COMP_AS_ING_OC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTIGUOUS_TO: [COMP_OBJ], ARG_CONTROLLED: [COMP_OBJ]}},
								 {COMP_AS_ING_OC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTIGUOUS_TO: [COMP_OBJ], ARG_CONTROLLED: [COMP_OBJ]}}),
	"NOM-NP-ING":				({COMP_ING_NPC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTIGUOUS_TO: COMP_NP, ARG_CONTROLLED: [COMP_NP]}},
								 {COMP_ING_NPC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTIGUOUS_TO: COMP_NP, ARG_CONTROLLED: [COMP_NP]}}),
	"NOM-NP-ING-OC":			({COMP_ING_OC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTIGUOUS_TO: COMP_OBJ, ARG_CONTROLLED: [COMP_OBJ]}},
								 {COMP_ING_OC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTIGUOUS_TO: COMP_OBJ, ARG_CONTROLLED: [COMP_OBJ]}}),
	"NOM-NP-ING-SC":			({COMP_ING_SC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTIGUOUS_TO: COMP_OBJ, ARG_CONTROLLED: [COMP_SUBJ]}},
								 {COMP_ING_SC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTIGUOUS_TO: COMP_OBJ, ARG_CONTROLLED: [COMP_SUBJ]}}),
	"NOM-NP-P-ING":				({COMP_P_ING_NPC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_NP]}},
								 {COMP_P_ING_NPC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_NP]}}),
	"NOM-NP-P-ING-OC":			({COMP_P_ING_OC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_OBJ]}},
								 {COMP_P_ING_OC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_OBJ]}}),
	"NOM-NP-P-ING-SC":			({COMP_P_ING_SC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_SUBJ]}},
								 {COMP_P_ING_SC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_SUBJ]}}),
	"NOM-ING-SC":				({COMP_ING_SC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_SUBJ]}, 						COMP_POSS_ING_SC: {ARG_CONTROLLED: [COMP_SUBJ], ARG_CONSTRAINTS: [ARG_CONSTRAINT_POSSESSIVE]}},
								 {COMP_ING_SC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_SUBJ]}}),
	"NOM-POSSING": 				({COMP_ING_POSSC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_POSS_ING_VC]}, 				COMP_POSS_ING_VC: {ARG_CONSTRAINTS: [ARG_CONSTRAINT_POSSESSIVE]}},
								 {COMP_ING_POSSC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_POSS_ING_VC]},				COMP_POSS_ING_VC: {ARG_CONSTRAINTS: [ARG_CONSTRAINT_POSSESSIVE]}}),
	"NOM-P-ING-SC": 			({COMP_P_ING_SC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_SUBJ]}, 						COMP_POSS_ING_SC: {ARG_CONTROLLED: [COMP_SUBJ], ARG_CONSTRAINTS: [ARG_CONSTRAINT_POSSESSIVE]}},
								 {COMP_P_ING_SC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_SUBJ]}}),
	"NOM-P-POSSING": 			({COMP_P_ING_POSSC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_POSS_ING_VC]}, 			COMP_POSS_ING_VC: {ARG_CONSTRAINTS: [ARG_CONSTRAINT_POSSESSIVE]}},
								 {COMP_P_ING_POSSC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_POSS_ING_VC]}, 			COMP_POSS_ING_VC: {ARG_CONSTRAINTS: [ARG_CONSTRAINT_POSSESSIVE]}}),
	"NOM-PP-P-POSSING": 		({COMP_P_ING_POSSC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_POSS_ING_VC]}, 			COMP_POSS_ING_VC: {ARG_CONSTRAINTS: [ARG_CONSTRAINT_POSSESSIVE]}},
								 {COMP_P_ING_POSSC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_POSS_ING_VC]}, 			COMP_POSS_ING_VC: {ARG_CONSTRAINTS: [ARG_CONSTRAINT_POSSESSIVE]}}),
	"NOM-P-NP-ING": 			({COMP_P_NP_ING: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING]}},
								 {COMP_P_NP_ING: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING]}}),
	"NOM-POSSING-PP": 			({COMP_ING_POSSC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_POSS_ING_VC]}, 				COMP_POSS_ING_VC: {ARG_CONSTRAINTS: [ARG_CONSTRAINT_POSSESSIVE]}},
								 {COMP_ING_POSSC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_POSS_ING_VC]}, 				COMP_POSS_ING_VC: {ARG_CONSTRAINTS: [ARG_CONSTRAINT_POSSESSIVE]}}),
	"NOM-NP-P-POSSING": 		({COMP_P_ING_POSSC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_POSS_ING_VC]}, 			COMP_POSS_ING_VC: {ARG_CONSTRAINTS: [ARG_CONSTRAINT_POSSESSIVE]}},
								 {COMP_P_ING_POSSC: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_POSS_ING_VC]}, 			COMP_POSS_ING_VC: {ARG_CONSTRAINTS: [ARG_CONSTRAINT_POSSESSIVE]}}),
	"NOM-NP-P-NP-ING": 			({COMP_P_NP_ING: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING]}},
								 {COMP_P_NP_ING: {ARG_HEAD_POSTAGS: [POSTAG_VBG], ARG_HEAD_PATTERNS: [PATTERN_ING]}}),

	# INF
	"NOM-FOR-TO-INF": 			({COMP_TO_INF_FOR_OC: {ARG_HEAD_LINKS: LINKS_TO, ARG_HEAD_POSTAGS: [POSTAG_VB], ARG_CONTROLLED: [COMP_FOR_NP], ARG_CONTIGUOUS_TO: COMP_FOR_NP}},
								 {COMP_TO_INF_FOR_OC: {ARG_HEAD_LINKS: LINKS_TO, ARG_HEAD_POSTAGS: [POSTAG_VB], ARG_CONTROLLED: [COMP_FOR_NP], ARG_CONTIGUOUS_TO: COMP_FOR_NP}}),
	"NOM-NP-TO-INF-OC": 		({COMP_TO_INF_OC: {ARG_HEAD_LINKS: LINKS_TO, ARG_HEAD_POSTAGS: [POSTAG_VB], ARG_CONTROLLED: [COMP_OBJ]}},
								 {COMP_TO_INF_OC: {ARG_HEAD_LINKS: LINKS_TO, ARG_HEAD_POSTAGS: [POSTAG_VB], ARG_CONTROLLED: [COMP_OBJ]}}),
	"NOM-NP-TO-INF-SC": 		({COMP_TO_INF_SC: {ARG_HEAD_LINKS: LINKS_TO, ARG_HEAD_POSTAGS: [POSTAG_VB], ARG_CONTROLLED: [COMP_SUBJ]}},
								 {COMP_TO_INF_SC: {ARG_HEAD_LINKS: LINKS_TO, ARG_HEAD_POSTAGS: [POSTAG_VB], ARG_CONTROLLED: [COMP_SUBJ]}}),
	"NOM-NP-TO-INF-VC": 		({COMP_TO_INF_VC: {ARG_HEAD_LINKS: LINKS_TO, ARG_HEAD_POSTAGS: [POSTAG_VB], ARG_CONTROLLED: [COMP_SUBJ, COMP_OBJ]}},
								 {COMP_TO_INF_VC: {ARG_HEAD_LINKS: LINKS_TO, ARG_HEAD_POSTAGS: [POSTAG_VB], ARG_CONTROLLED: [COMP_SUBJ, COMP_OBJ], ARG_CONTIGUOUS_TO: COMP_OBJ}}), # ONLY FOR NOM
	"NOM-TO-INF-SC":			({COMP_TO_INF_SC: {ARG_HEAD_LINKS: LINKS_TO, ARG_HEAD_POSTAGS: [POSTAG_VB], ARG_CONTROLLED: [COMP_SUBJ]}},
								 {COMP_TO_INF_SC: {ARG_HEAD_LINKS: LINKS_TO, ARG_HEAD_POSTAGS: [POSTAG_VB], ARG_CONTROLLED: [COMP_SUBJ]}}),
	"NOM-P-NP-TO-INF-OC":		({COMP_TO_INF_P_OC: {ARG_HEAD_LINKS: LINKS_TO, ARG_HEAD_POSTAGS: [POSTAG_VB], ARG_CONTROLLED: [COMP_PP]}},
								 {COMP_TO_INF_P_OC: {ARG_HEAD_LINKS: LINKS_TO, ARG_HEAD_POSTAGS: [POSTAG_VB], ARG_CONTROLLED: [COMP_PP]}}),
	"NOM-P-NP-TO-INF":			({COMP_TO_INF_P_OC: {ARG_HEAD_LINKS: LINKS_TO, ARG_HEAD_POSTAGS: [POSTAG_VB], ARG_CONTROLLED: [COMP_PP], ARG_CONTIGUOUS_TO: COMP_PP}},
								 {COMP_TO_INF_P_OC: {ARG_HEAD_LINKS: LINKS_TO, ARG_HEAD_POSTAGS: [POSTAG_VB], ARG_CONTROLLED: [COMP_PP], ARG_CONTIGUOUS_TO: COMP_PP}}),
	"NOM-P-NP-TO-INF-VC":		({COMP_TO_INF_VC: {ARG_HEAD_LINKS: LINKS_TO, ARG_HEAD_POSTAGS: [POSTAG_VB], ARG_CONTROLLED: [COMP_SUBJ, COMP_PP], ARG_CONTIGUOUS_TO: [COMP_PP]}},
								 {COMP_TO_INF_VC: {ARG_HEAD_LINKS: LINKS_TO, ARG_HEAD_POSTAGS: [POSTAG_VB], ARG_CONTROLLED: [COMP_SUBJ, COMP_PP], ARG_CONTIGUOUS_TO: [COMP_PP]}}),
	"NOM-PP-FOR-TO-INF":		({COMP_TO_INF_FOR_OC: {ARG_HEAD_LINKS: LINKS_TO, ARG_HEAD_POSTAGS: [POSTAG_VB], ARG_CONTROLLED: [COMP_FOR_NP]}},
								 {COMP_TO_INF_FOR_OC: {ARG_HEAD_LINKS: LINKS_TO, ARG_HEAD_POSTAGS: [POSTAG_VB], ARG_CONTROLLED: [COMP_FOR_NP]}}),

	"NOM-S":					({COMP_SBAR: {}}, # The "that" complementizer word isn't required for verbs
								 {COMP_SBAR: {ARG_HEAD_LINKS: LINKS_THAT}}),
	"NOM-THAT-S":				({COMP_SBAR: {ARG_HEAD_LINKS: LINKS_THAT}},
								 {COMP_SBAR: {ARG_HEAD_LINKS: LINKS_THAT}}),
	"NOM-NP-S":					({COMP_SBAR: {}}, # The "that" complementizer word isn't required for verbs
								 {COMP_SBAR: {ARG_HEAD_LINKS: LINKS_THAT}}),
	"NOM-PP-THAT-S":			({COMP_SBAR: {ARG_HEAD_LINKS: LINKS_THAT}},
								 {COMP_SBAR: {ARG_HEAD_LINKS: LINKS_THAT}}),

	# Illegal positions
	"NOM-HOW-S":				({COMP_HOW_S: {ARG_ILLEGAL_PREFIXES: ["how to", "how many", "how much"]}},
								 {COMP_HOW_S: {ARG_ILLEGAL_PREFIXES: ["how to", "how many", "how much"]}}),

	# SUBJUNCT
	"NOM-S-SUBJUNCT":			({COMP_SBAR: {ARG_SUBJUNCT: True}}, # The "that" complementizer word isn't required for verbs
								 {COMP_SBAR: {ARG_SUBJUNCT: True, ARG_HEAD_LINKS: LINKS_THAT}}),
	"NOM-PP-THAT-S-SUBJUNCT":	({COMP_SBAR: {ARG_SUBJUNCT: True, ARG_HEAD_LINKS: LINKS_THAT}},
								 {COMP_SBAR: {ARG_SUBJUNCT: True, ARG_HEAD_LINKS: LINKS_THAT}}),
	"NOM-NP-AS-IF-S-SUBJUNCT":	({COMP_AS_IF_S: {ARG_SUBJUNCT: True}},
								 {COMP_AS_IF_S: {ARG_SUBJUNCT: True}})
}


# Dictionary that includes some subcategorization typo mistakes that appear in the lexicon
# {original_mistaken_subcat: right_subcat}
subcat_typos_dict = {
	"NOM-INSTRANS": "NOM-INTRANS",
	"INTRANS": "NOM-INTRANS"
}

# Dictionary that includes the possible "translations" for each nom-type into complement
# {NOM_TYPE: [COMP1, COMP2]}
nom_types_to_args_dict = {
	NOM_TYPE_VERB_NOM: [],
	NOM_TYPE_IND_OBJ: [COMP_IND_OBJ, COMP_PP1, COMP_PP2, COMP_PP],
	NOM_TYPE_SUBJ: [COMP_SUBJ],
	NOM_TYPE_OBJ: [COMP_OBJ],
	NOM_TYPE_P_OBJ: [COMP_PP, COMP_P_IND_OBJ],
	NOM_TYPE_INSTRUMENT: [COMP_INSTRUMENT],
}