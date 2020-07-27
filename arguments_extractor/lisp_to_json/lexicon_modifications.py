from arguments_extractor.constants.lexicon_constants import *
from arguments_extractor.constants.ud_constants import *

# This file includes several "tables" (dictionary) for needed modifications for the lexicon, based on the manual



# Dictionary that includes fixes for specific subcats, based on the manual
# The subcats in this dictionary are assumed to be "the known subcats"
# {SUBCAT: ((VERB_REQUIRED, VERB_DEFAULTS, VERB_TRANSLATION, VERB_SPECIAL_VALUES),
# 		    (NOM_REQUIRED, NOM_OPTIONAL, NOM_TRANSLATIONS, NOM_SPECIAL_VALUES))}
lexicon_fixes_dict = {
	"NOM-INTRANS": 				(([], 										{},																		{},													{}),
								 ([], 										{},																		{},													{})),
	"NOM-INTRANS-RECIP": 		(([], 										{},																		{},													{}),
								 ([], 										{}, 																	{},													{})),


	# NP and PP
	"NOM-NP": 					(([COMP_OBJ], 								{}, 																	{},													{}),
								 ([], 										{}, 																	{},													{})),
	"NOM-NP-NP": 				(([COMP_IND_OBJ, COMP_OBJ],					{COMP_IND_OBJ: [POS_IOBJ]}, 											{COMP_IND_OBJ: IGNORE_COMP},						{}),
								 ([COMP_IND_OBJ, COMP_OBJ], 				{},																		{},													{})),
	"NOM-PP": 					(([COMP_PP], 								{}, 																	{},													{}),
								 ([COMP_PP], 								{}, 																	{},													{})),
	"NOM-PP-PP": 				(([COMP_PP1, COMP_PP2], 					{}, 																	{COMP_PP: COMP_PP1},								{}),
								 ([COMP_PP1, COMP_PP2], 					{},																		{COMP_PP: COMP_PP1},								{})),
	"NOM-NP-PP": 				(([COMP_OBJ, COMP_PP], 						{}, 																	{},													{}),
								 ([COMP_PP],								{},																		{},													{})),
	"NOM-NP-TO-NP": 			(([COMP_OBJ, COMP_IND_OBJ],					{COMP_IND_OBJ: ["to", POS_IOBJ]},										{COMP_IND_OBJ: IGNORE_COMP},						{}),
								 ([COMP_OBJ, COMP_IND_OBJ],					{COMP_IND_OBJ: ["to"]},													{},													{})),
	"NOM-NP-FOR-NP": 			(([COMP_OBJ, COMP_IND_OBJ],					{COMP_IND_OBJ: ["for", POS_IOBJ]},										{COMP_IND_OBJ: IGNORE_COMP},						{}),
								 ([COMP_OBJ, COMP_IND_OBJ],					{COMP_IND_OBJ: ["for"]},												{},													{})),
	"NOM-NP-PP-PP": 			(([COMP_OBJ, COMP_PP1, COMP_PP2],			{},																		{COMP_PP: COMP_PP1},								{}),
								 ([COMP_PP1, COMP_PP2],						{},																		{COMP_PP: COMP_PP1},								{})),


	# ADVP and ADJP
	"NOM-ADVP": 				(([COMP_ADVP], 								{COMP_ADVP: [POS_ADMOD]}, 												{},													{}),
								 ([], 										{COMP_ADVP: [POS_ADMOD, OPT_POS], COMP_ADJP: [POS_AJMOD, OPT_POS]},		{},													{})),
	"NOM-NP-ADVP": 				(([COMP_OBJ, COMP_ADVP], 					{COMP_ADVP: [POS_ADMOD]}, 												{},													{}),
								 ([],										{COMP_ADVP: [POS_ADMOD, OPT_POS], COMP_ADJP: [POS_AJMOD, OPT_POS]}, 	{},													{})),
	"NOM-ADVP-PP": 				(([COMP_ADVP, COMP_PP], 					{COMP_ADVP: [POS_ADMOD]}, 												{},													{}),
								 ([COMP_PP],								{COMP_ADVP: [POS_ADMOD, OPT_POS], COMP_ADJP: [POS_AJMOD, OPT_POS]},		{},													{})),


	# AS PHRASE
	"NOM-NP-AS-ING":	 		(([COMP_OBJ, COMP_AS_ING_OC], 				{COMP_AS_ING_OC: ["as"]}, 												{},													{}),
								 ([COMP_OBJ, COMP_AS_ING_OC], 				{COMP_AS_ING_OC: ["as"]},												{},													{})),
	"NOM-NP-AS-ADJP": 			(([COMP_OBJ, COMP_AS_ADJP], 				{COMP_AS_ADJP: ["as"]}, 												{},													{}),
								 ([COMP_OBJ, COMP_AS_ADJP], 				{COMP_AS_ADJP: ["as"]},													{},													{})),
	"NOM-AS-NP": 				(([COMP_SUBJ, COMP_AS_NP_SC], 				{COMP_AS_NP_SC: ["as"]}, 												{},													{}),
								 ([COMP_SUBJ, COMP_AS_NP_SC], 				{COMP_AS_NP_SC: ["as"]},												{},													{})),
	"NOM-NP-AS-NP": 			(([COMP_OBJ, COMP_AS_NP_OC], 				{COMP_AS_NP_OC: ["as"]}, 												{},													{}),
								 ([COMP_OBJ, COMP_AS_NP_OC],				{COMP_AS_NP_OC: ["as"]},												{},													{})),
	"NOM-NP-AS-NP-SC": 			(([COMP_SUBJ, COMP_OBJ, COMP_AS_NP_SC], 	{COMP_AS_NP_SC: ["as"]}, 												{},													{}),
								 ([COMP_SUBJ, COMP_AS_NP_SC],				{COMP_AS_NP_SC: ["as"]},												{},													{})),
	"NOM-NP-PP-AS-NP": 			(([COMP_OBJ, COMP_PP, COMP_AS_NP_OC], 		{COMP_AS_NP_OC: ["as"]}, 												{COMP_PP1: COMP_PP},								{}),
								 ([COMP_OBJ, COMP_PP, COMP_AS_NP_OC],		{COMP_AS_NP_OC: ["as"]},												{COMP_PP1: COMP_PP},								{})),


	# GERUNDS
	"NOM-ING-SC": 				(([COMP_ING_SC], 							{COMP_ING_SC: [POS_ING]}, 												{},													{}),
								 ([COMP_ING_SC],							{COMP_ING_SC: ["of"]},													{},													{COMP_ING_SC: ["NOM-SUBC", "P-", "PVAL"]})), # NOM-SUBC just for nom
	"NOM-NP-ING": 				(([COMP_NP, COMP_ING_NPC],					{}, 																	{COMP_OBJ: IGNORE_COMP},							{}),
								 ([COMP_NP, COMP_ING_NPC],					{},																		{COMP_OBJ: COMP_NP},								{})),
	"NOM-NP-ING-OC": 			(([COMP_OBJ, COMP_ING_OC], 					{},																		{},													{}),
								 ([COMP_OBJ, COMP_ING_OC],					{},																		{},													{})),
	"NOM-NP-ING-SC": 			(([COMP_OBJ, COMP_ING_SC], 					{}, 																	{},													{}),
								 ([COMP_OBJ, COMP_ING_SC],					{},																		{},													{})),
	"NOM-P-ING-SC":				(([COMP_P_ING_SC],							{},																		{COMP_PP: COMP_P_ING_SC},							{COMP_P_ING_SC: ["NOM-SUBC", "P-", "PVAL"]}),
								 ([COMP_P_ING_SC],							{},																		{COMP_PP: COMP_P_ING_SC},							{COMP_P_ING_SC: ["NOM-SUBC", "P-", "PVAL"]})), # NOM-SUBC for both
	"NOM-NP-P-ING": 			(([COMP_NP, COMP_P_ING_NPC],				{},																		{COMP_PP: COMP_P_ING_NPC, COMP_OBJ: COMP_NP},		{COMP_P_ING_NPC: ["NOM-SUBC", "P-", "PVAL"]}),
								 ([COMP_NP, COMP_P_ING_NPC], 				{}, 																	{COMP_PP: COMP_P_ING_NPC, COMP_OBJ: COMP_NP}, 		{COMP_P_ING_NPC: ["NOM-SUBC", "P-", "PVAL"]})), # NOM-SUBC for both
	"NOM-NP-P-ING-OC": 			(([COMP_OBJ, COMP_P_ING_OC], 				{}, 																	{COMP_PP: COMP_P_ING_OC}, 							{COMP_P_ING_OC: ["NOM-SUBC", "P-", "PVAL"]}),
								 ([COMP_OBJ, COMP_P_ING_OC], 				{}, 																	{COMP_PP: COMP_P_ING_OC}, 							{COMP_P_ING_OC: ["NOM-SUBC", "P-", "PVAL"]})), # NOM-SUBC for both
	"NOM-NP-P-ING-SC": 			(([COMP_OBJ, COMP_P_ING_SC],				{}, 																	{COMP_PP: COMP_P_ING_SC}, 							{COMP_P_ING_SC: ["NOM-SUBC", "P-", "PVAL"]}),
								 ([COMP_OBJ, COMP_P_ING_SC],	 			{}, 																	{COMP_PP: COMP_P_ING_SC},							{COMP_P_ING_SC: ["NOM-SUBC", "P-", "PVAL"]})), # NOM-SUBC for both
	"NOM-P-NP-ING": 			(([COMP_PP, COMP_ING_POC],					{},																		{},													{}),
								 ([COMP_PP, COMP_ING_POC], 					{},	 																	{},													{})),
	"NOM-NP-P-NP-ING": 			(([COMP_OBJ, COMP_PP, COMP_ING_POC],		{},																		{},													{}),
								 ([COMP_OBJ, COMP_PP, COMP_ING_POC],		{},																		{},													{})),


	# POSSESIVE GERUNDS
	"NOM-POSSING": 				(([COMP_POSSING], 							{COMP_POSSING: [POS_ING]}, 												{},													{}),
								 ([COMP_POSSING],							{COMP_POSSING: ["of"]},													{},													{COMP_POSSING: ["NOM-SUBC", "P-", "PVAL"]})), # NOM-SUBC just for nom
	"NOM-P-POSSING": 			(([COMP_P_POSSING], 						{},																		{COMP_PP: COMP_P_POSSING},							{COMP_P_POSSING: ["NOM-SUBC", "P-", "PVAL"]}),
								 ([COMP_P_POSSING], 						{}, 																	{COMP_PP: COMP_P_POSSING}, 							{COMP_P_POSSING: ["NOM-SUBC", "P-", "PVAL"]})), # NOM-SUBC for both
	"NOM-PP-P-POSSING": 		(([COMP_PP, COMP_P_POSSING], 				{COMP_P_POSSING: ["about","on"], COMP_PP: ["between","among","with"]}, 	{COMP_PP: COMP_P_POSSING, COMP_PP1: COMP_PP},		{}),
								 ([COMP_PP, COMP_P_POSSING],				{COMP_P_POSSING: ["about","on"], COMP_PP: ["between","among","with"]},	{COMP_PP: COMP_P_POSSING, COMP_PP1: COMP_PP},		{})),
	"NOM-POSSING-PP": 			(([COMP_POSSING, COMP_PP], 					{COMP_POSSING: [POS_ING]}, 												{},													{}),
								 ([COMP_POSSING, COMP_PP],					{COMP_POSSING: ["of"]},													{},													{COMP_POSSING: ["NOM-SUBC", "P-", "PVAL"]})), # NOM-SUBC just for nom
	"NOM-NP-P-POSSING": 		(([COMP_OBJ, COMP_P_POSSING],				{},																		{COMP_PP: COMP_P_POSSING},							{COMP_P_POSSING: ["NOM-SUBC", "P-", "PVAL"]}),
								 ([COMP_OBJ, COMP_P_POSSING], 				{}, 																	{COMP_PP: COMP_P_POSSING}, 							{COMP_P_POSSING: ["NOM-SUBC", "P-", "PVAL"]})), # NOM-SUBC for both


	# INFINITIVE
	"NOM-FOR-TO-INF": 			(([COMP_FOR_TO_INF],						{COMP_FOR_TO_INF: ["for", POS_TO_INF]},									{},													{}),
								 ([COMP_FOR_TO_INF],						{COMP_FOR_TO_INF: ["for", POS_TO_INF]},									{},													{})),
	#"NOM-NP-TO-INF":			(([COMP_OBJ, FOR_TO_INF_OR_TO_INF], 		{COMP_TO_INF: [POS_TO_INF], COMP_FOR_NP_TO_INF: ["for"]},				{},													{}),
	#					 		 ([COMP_OBJ, FOR_TO_INF_OR_TO_INF], 		{COMP_TO_INF: [POS_TO_INF], COMP_FOR_NP_TO_INF: ["for"]},				{}, 												{})), # {COMP_TO_INF: ["NOM-SUBC", "FOR-TO-INF"]}), # Not appear in the manual
	"NOM-NP-TO-INF-OC": 		(([COMP_OBJ, COMP_TO_INF_OC], 				{COMP_TO_INF_OC: [POS_TO_INF]}, 										{},													{}),
								 ([COMP_OBJ, COMP_TO_INF_OC],				{COMP_TO_INF_OC: [POS_TO_INF]},											{},													{})),
	"NOM-NP-TO-INF-SC": 		(([COMP_OBJ, COMP_TO_INF_SC], 				{COMP_TO_INF_SC: [POS_TO_INF]}, 										{},													{}),
								 ([COMP_OBJ, COMP_TO_INF_SC],				{COMP_TO_INF_SC: [POS_TO_INF]},											{},													{})),
	"NOM-NP-TO-INF-VC": 		(([COMP_OBJ, COMP_TO_INF_VC], 				{COMP_TO_INF_VC: [POS_TO_INF]}, 										{},													{}),
								 ([COMP_OBJ, COMP_TO_INF_VC],				{COMP_TO_INF_VC: [POS_TO_INF]},											{},													{})),
	"NOM-TO-INF-SC": 			(([COMP_TO_INF_SC], 						{COMP_TO_INF_SC: [POS_TO_INF]}, 										{},													{}),
								 ([COMP_TO_INF_SC],							{COMP_TO_INF_SC: [POS_TO_INF]},											{},													{})),
	"NOM-P-NP-TO-INF-OC":		(([COMP_PP, COMP_TO_INF_POC], 				{COMP_TO_INF_POC: [POS_TO_INF]}, 										{},													{}),
								 ([COMP_PP, COMP_TO_INF_POC],				{COMP_TO_INF_POC: [POS_TO_INF]},										{},													{})),
	"NOM-P-NP-TO-INF":			(([COMP_PP, COMP_TO_INF_POC], 				{COMP_TO_INF_POC: [POS_TO_INF]}, 										{},													{}),
								 ([COMP_PP, COMP_TO_INF_POC],				{COMP_TO_INF_POC: [POS_TO_INF]},										{},													{})), # why is it different from NOM-P-NP-TO-INF-OC ?????
	"NOM-P-NP-TO-INF-VC":		(([COMP_PP, COMP_TO_INF_VC], 				{COMP_TO_INF_VC: [POS_TO_INF]}, 										{},													{}),
								 ([COMP_PP, COMP_TO_INF_VC],				{COMP_TO_INF_VC: [POS_TO_INF]},											{},													{})),
	"NOM-PP-FOR-TO-INF":		(([COMP_PP, COMP_FOR_TO_INF],				{COMP_FOR_TO_INF: ["for", POS_TO_INF]},									{COMP_PP1: COMP_PP},								{}),
								 ([COMP_PP, COMP_FOR_TO_INF],				{COMP_FOR_TO_INF: ["for", POS_TO_INF]},									{COMP_PP1: COMP_PP},								{})),
	# "NOM-PP-TO-INF-RECIP":		(([COMP_TO_INF_SC],	 						{COMP_SECOND_SUBJ: ["with", OPT_POS], COMP_TO_INF_SC: [POS_TO_INF]},	{COMP_PP1: COMP_SECOND_SUBJ},						{}),
	# 							 ([COMP_TO_INF_SC, COMP_SECOND_SUBJ], 		{COMP_SECOND_SUBJ: ["with"], COMP_TO_INF_SC: [POS_TO_INF]},				{COMP_PP1: COMP_SECOND_SUBJ},						{})), # SUBJECT = SUBJECT + P-NP (with). TOO RARE


	# SBAR
	"NOM-S":					(([COMP_SBAR],								{COMP_SBAR: [POS_SBAR, POS_THAT]}, 										{},													{}),
								 ([COMP_SBAR], 								{COMP_SBAR: [POS_THAT]},												{},													{})),
	"NOM-THAT-S":				(([COMP_SBAR],								{COMP_SBAR: [POS_THAT]}, 												{},													{}),
								 ([COMP_SBAR], 								{COMP_SBAR: [POS_THAT]},												{},													{})),
	"NOM-S-SUBJUNCT":			(([COMP_SBAR_SUBJUNCT],						{COMP_SBAR_SUBJUNCT: [POS_SBAR, POS_THAT]},								{},													{}),
								 ([COMP_SBAR_SUBJUNCT], 					{COMP_SBAR_SUBJUNCT: [POS_THAT]},										{},													{})), # Means that the verb of SBAR is in subjunctive mood
	"NOM-NP-S":					(([COMP_OBJ, COMP_SBAR],					{COMP_SBAR: [POS_SBAR, POS_THAT]},										{},													{}),
								 ([COMP_OBJ, COMP_SBAR], 					{COMP_SBAR: [POS_THAT]},												{},													{})),
	"NOM-PP-THAT-S":			(([COMP_PP, COMP_SBAR],						{COMP_SBAR: [POS_THAT]},												{COMP_PP1: COMP_PP},								{}),
								 ([COMP_PP, COMP_SBAR], 					{COMP_SBAR: [POS_THAT]},												{COMP_PP1: COMP_PP},								{})),
	"NOM-PP-THAT-S-SUBJUNCT":	(([COMP_PP, COMP_SBAR_SUBJUNCT],			{COMP_SBAR_SUBJUNCT: [POS_THAT]},										{COMP_PP1: COMP_PP},								{}),
								 ([COMP_PP, COMP_SBAR_SUBJUNCT],			{COMP_SBAR_SUBJUNCT: [POS_THAT]},										{COMP_PP1: COMP_PP},								{})),
	"NOM-NP-AS-IF-S-SUBJUNCT":	(([COMP_OBJ, COMP_AS_IF_S_SUBJUNCT],		{COMP_AS_IF_S_SUBJUNCT: ["as if"]},										{},													{}),
								 ([COMP_OBJ, COMP_AS_IF_S_SUBJUNCT], 		{COMP_AS_IF_S_SUBJUNCT: ["as if"]},										{},													{})),


	# WH-SBAR
	# There must be a WH-word after the preposition that preceds the WH-phase. The WH-word is determined based on the complement's name:
	# - WH-S, P-WH-S: if (only for verb), whether, what
	# - WHERE-WHERN-S: where, when, how-quant-n (how many and how much)
	# - HOW-S: how
	# - HOW-TO-INF: how to
	# If where-when tag appears under P-WH in NOM-SUBC, than there are probably more possible tags for both nom and verb [?????]. This occurs only for NOM-WH-S.
	"NOM-WH-S":					(([COMP_WH_S],								{COMP_WH_S: WH_VERB_OPTIONS},											{},													{}),
								 ([COMP_WH_S], 								{COMP_WH_S: ["of"]},													{},													{COMP_WH_S: ["NOM-SUBC", "P-WH", "PVAL"]})), # NOM-SUBC for just nom
	"NOM-WHERE-WHEN-S":			(([COMP_WHERE_WHEN_S],						{COMP_WHERE_WHEN_S: WHERE_WHEN_OPTIONS},								{},													{}),
								 ([COMP_WHERE_WHEN_S], 						{COMP_WHERE_WHEN_S: ["of"]},											{},													{COMP_WHERE_WHEN_S: ["NOM-SUBC", "P-WH", "PVAL"]})), # NOM-SUBC for just nom
	"NOM-HOW-S":				(([COMP_HOW_S],								{COMP_HOW_S: HOW_OPTIONS},												{},													{}),
								 ([COMP_HOW_S], 							{COMP_HOW_S: ["of"]},													{},													{COMP_HOW_S: ["NOM-SUBC", "P-WH", "PVAL"]})), # NOM-SUBC for just nom
	"NOM-PP-HOW-TO-INF":		(([COMP_HOW_TO_INF],						{COMP_HOW_TO_INF: HOW_TO_OPTIONS, COMP_PP: [OPT_POS]},					{COMP_PP1: COMP_PP},								{}),
								 ([COMP_HOW_TO_INF],						{COMP_HOW_TO_INF: ["of"], COMP_PP: [OPT_POS]},							{COMP_PP1: COMP_PP},								{COMP_HOW_TO_INF: ["NOM-SUBC", "P-WH", "PVAL"]})), # NOM-SUBC for just nom
	"NOM-NP-WH-S":				(([COMP_OBJ, COMP_WH_S],					{COMP_WH_S: WH_VERB_OPTIONS},											{},													{}),
								 ([COMP_WH_S], 								{COMP_WH_S: ["of"]},													{},													{COMP_WH_S: ["NOM-SUBC", "P-WH", "PVAL"]})), # NOM-SUBC for just nom
	"NOM-P-WH-S":				(([COMP_P_WH_S], 							{}, 																	{COMP_PP: COMP_P_WH_S}, 							{COMP_P_WH_S: ["NOM-SUBC", "P-WH", "PVAL"]}),
								 ([COMP_P_WH_S], 							{},						 												{COMP_PP: COMP_P_WH_S}, 							{COMP_P_WH_S: ["NOM-SUBC", "P-WH", "PVAL"]})), # NOM-SUBC for both
	"NOM-PP-WH-S":				(([COMP_PP, COMP_WH_S],						{COMP_WH_S: WH_VERB_OPTIONS},											{COMP_PP1: COMP_PP},								{}),
								 ([COMP_PP, COMP_WH_S],						{COMP_WH_S: ["of"]},													{COMP_PP1: COMP_PP},								{COMP_WH_S: ["NOM-SUBC", "P-WH", "PVAL"]})), # NOM-SUBC for just nom
	"NOM-PP-P-WH-S":			(([COMP_PP, COMP_P_WH_S], 					{}, 																	{COMP_PP: COMP_P_WH_S, COMP_PP1: COMP_PP}, 			{COMP_P_WH_S: ["NOM-SUBC", "P-WH", "PVAL"]}),
								 ([COMP_PP, COMP_P_WH_S], 					{},	 																	{COMP_PP: COMP_P_WH_S, COMP_PP1: COMP_PP}, 			{COMP_P_WH_S: ["NOM-SUBC", "P-WH", "PVAL"]})), # NOM-SUBC for both
	"NOM-NP-P-WH-S":			(([COMP_OBJ, COMP_P_WH_S], 					{}, 																	{COMP_PP: COMP_P_WH_S}, 							{COMP_P_WH_S: ["NOM-SUBC", "P-WH", "PVAL"]}),
								 ([COMP_P_WH_S], 							{}, 																	{COMP_PP: COMP_P_WH_S}, 							{COMP_P_WH_S: ["NOM-SUBC", "P-WH", "PVAL"]})), # NOM-SUBC for both
}



# Dicationary that includes a more complex argument positions structure, similarly to the default column in the above dictionary
# {SUBCAT: (VERB_CONSTRAINTS, NOM_CONSTRAINTS)}
complex_argument_positions = {
	"NOM-NP-ING":				({COMP_NP: [{COMP_ING_NPC: [POS_NSUBJ, POS_DET_POSS]}], 	COMP_ING_NPC: [POS_ING]},
								 {COMP_ING_NPC: [POS_ING]}),
	"NOM-NP-ING-OC":			({COMP_ING_OC: [POS_ING, {COMP_OBJ: [POS_ACL]}]},
								 {COMP_ING_OC: [POS_ING]}),
	"NOM-NP-ING-SC":			({COMP_ING_SC: [POS_ING]},
								 {COMP_ING_SC: [POS_ING]}),
	"NOM-P-NP-ING":				({COMP_ING_POC: [POS_ING, {COMP_PP: [POS_ACL]}], 			COMP_PP: [ADD_POS, {COMP_ING_POC: [POS_NSUBJ]}]},
								 {COMP_ING_POC: [POS_ING, {COMP_PP: [POS_ACL]}], 			COMP_PP: [ADD_POS, {COMP_ING_POC: [POS_NSUBJ]}]}),
	"NOM-NP-P-NP-ING":			({COMP_ING_POC: [POS_ING, {COMP_PP: [POS_ACL]}], 			COMP_PP: [ADD_POS, {COMP_ING_POC: [POS_NSUBJ]}]},
								 {COMP_ING_POC: [POS_ING, {COMP_PP: [POS_ACL]}], 			COMP_PP: [ADD_POS, {COMP_ING_POC: [POS_NSUBJ]}]}),
}



# Dicationary that includes a list of constraints for specific subcats
# {SUBCAT: (VERB_CONSTRAINTS, NOM_CONSTRAINTS)}
subcat_constraints = {
	"NOM-ADVP":					([],
								 [SUBCAT_CONSTRAINT_ADVP_OR_ADJP]),
	"NOM-NP-ADVP":				([],
								 [SUBCAT_CONSTRAINT_ADVP_OR_ADJP]),
	"NOM-ADVP-PP":				([],
								 [SUBCAT_CONSTRAINT_ADVP_OR_ADJP]),
}



# Dicationary that includes a list of constraints for arguments of specific subcats
# {SUBCAT: (VERB_CONSTRAINTS, NOM_CONSTRAINTS)}
argument_constraints = {
	# PLURAL and INCLUDING
	"NOM-INTRANS-RECIP": 		({COMP_SUBJ: {ARG_CONSTRAINTS: [ARG_CONSTRAINT_PLURAL]}},
								 {COMP_SUBJ: {ARG_CONSTRAINTS: [ARG_CONSTRAINT_PLURAL]}}),
	"NOM-PP-TO-INF-RECIP":		({COMP_SUBJ: {ARG_INCLUDING: COMP_SECOND_SUBJ, ARG_CONSTRAINTS: [ARG_CONSTRAINT_PLURAL]}, 			COMP_TO_INF_SC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_URELATIONS: [URELATION_TO], ARG_CONTROLLED: [COMP_SUBJ]}},
								 {COMP_SUBJ: {ARG_INCLUDING: COMP_SECOND_SUBJ, ARG_CONSTRAINTS: [ARG_CONSTRAINT_PLURAL]}, 			COMP_TO_INF_SC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_URELATIONS: [URELATION_TO], ARG_CONTROLLED: [COMP_SUBJ]}}),

	"NOM-NP-NP":				({},
								 {COMP_OBJ: {ARG_CONTIGUOUS_TO: COMP_IND_OBJ}}), # ONLY FOR VERB

	# AS-NP
	"NOM-AS-NP":				({COMP_AS_NP_SC: {ARG_CONTROLLED: [COMP_SUBJ]}},
								 {COMP_AS_NP_SC: {ARG_CONTROLLED: [COMP_SUBJ]}}),
	"NOM-NP-AS-NP":				({COMP_AS_NP_OC: {ARG_CONTROLLED: [COMP_OBJ]}},
								 {COMP_AS_NP_OC: {ARG_CONTROLLED: [COMP_OBJ]}}),
	"NOM-NP-AS-NP-SC":			({COMP_AS_NP_SC: {ARG_CONTROLLED: [COMP_SUBJ]}},
								 {COMP_AS_NP_SC: {ARG_CONTROLLED: [COMP_SUBJ]}}),
	"NOM-NP-PP-AS-NP":			({COMP_AS_NP_OC: {ARG_CONTROLLED: [COMP_OBJ]}},
								 {COMP_AS_NP_OC: {ARG_CONTROLLED: [COMP_OBJ]}}),

	# ADVP AND ADJP
	"NOM-ADVP":					({COMP_ADVP: {ARG_ROOT_UPOSTAGS: [UPOS_ADV]}},
								 {COMP_ADVP: {ARG_ROOT_UPOSTAGS: [UPOS_ADV]}, 				COMP_ADJP: {ARG_ROOT_UPOSTAGS: [UPOS_ADJ]}}),
	"NOM-NP-ADVP":				({COMP_ADVP: {ARG_ROOT_UPOSTAGS: [UPOS_ADV]}},
								 {COMP_ADVP: {ARG_ROOT_UPOSTAGS: [UPOS_ADV]}, 				COMP_ADJP: {ARG_ROOT_UPOSTAGS: [UPOS_ADJ]}}),
	"NOM-ADVP-PP":				({COMP_ADVP: {ARG_ROOT_UPOSTAGS: [UPOS_ADV]}},
								 {COMP_ADVP: {ARG_ROOT_UPOSTAGS: [UPOS_ADV]}, 				COMP_ADJP: {ARG_ROOT_UPOSTAGS: [UPOS_ADJ]}}),
	"NOM-NP-AS-ADJP":			({COMP_AS_ADJP: {ARG_ROOT_UPOSTAGS: [UPOS_ADJ]}},
								 {COMP_AS_ADJP: {ARG_ROOT_UPOSTAGS: [UPOS_ADJ]}}),

	# ING
	"NOM-NP-AS-ING":			({COMP_AS_ING_OC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONTIGUOUS_TO: [COMP_OBJ], ARG_CONTROLLED: [COMP_OBJ]}},
								 {COMP_AS_ING_OC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONTIGUOUS_TO: [COMP_OBJ], ARG_CONTROLLED: [COMP_OBJ]}}),
	"NOM-NP-ING":				({COMP_ING_NPC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONTIGUOUS_TO: COMP_NP, ARG_CONTROLLED: [COMP_NP]}},
								 {COMP_ING_NPC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONTIGUOUS_TO: COMP_NP, ARG_CONTROLLED: [COMP_NP]}}),
	"NOM-NP-ING-OC":			({COMP_ING_OC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONTIGUOUS_TO: COMP_OBJ, ARG_CONTROLLED: [COMP_OBJ]}},
								 {COMP_ING_OC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONTIGUOUS_TO: COMP_OBJ, ARG_CONTROLLED: [COMP_OBJ]}}),
	"NOM-NP-ING-SC":			({COMP_ING_SC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONTIGUOUS_TO: COMP_OBJ, ARG_CONTROLLED: [COMP_SUBJ]}},
								 {COMP_ING_SC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONTIGUOUS_TO: COMP_OBJ, ARG_CONTROLLED: [COMP_SUBJ]}}),
	"NOM-NP-P-ING":				({COMP_P_ING_NPC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_NP]}},
								 {COMP_P_ING_NPC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_NP]}}),
	"NOM-NP-P-ING-OC":			({COMP_P_ING_OC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_OBJ]}},
								 {COMP_P_ING_OC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_OBJ]}}),
	"NOM-NP-P-ING-SC":			({COMP_P_ING_SC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_SUBJ]}},
								 {COMP_P_ING_SC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_SUBJ]}}),
	"NOM-ING-SC":				({COMP_ING_SC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_SUBJ], ARG_CONSTRAINTS: [ARG_CONSTRAINT_OPTIONAL_POSSESSIVE]}},
								 {COMP_ING_SC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_SUBJ]}}),
	"NOM-POSSING": 				({COMP_POSSING: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONSTRAINTS: [ARG_CONSTRAINT_OPTIONAL_POSSESSIVE]}},
								 {COMP_POSSING: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONSTRAINTS: [ARG_CONSTRAINT_OPTIONAL_POSSESSIVE]}}),
	"NOM-P-ING-SC": 			({COMP_P_ING_SC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_SUBJ], ARG_CONSTRAINTS: [ARG_CONSTRAINT_OPTIONAL_POSSESSIVE]}},
								 {COMP_P_ING_SC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_SUBJ]}}),
	"NOM-P-POSSING": 			({COMP_P_POSSING: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONSTRAINTS: [ARG_CONSTRAINT_OPTIONAL_POSSESSIVE]}},
								 {COMP_P_POSSING: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONSTRAINTS: [ARG_CONSTRAINT_OPTIONAL_POSSESSIVE]}}),
	"NOM-PP-P-POSSING": 		({COMP_P_POSSING: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONSTRAINTS: [ARG_CONSTRAINT_OPTIONAL_POSSESSIVE]}},
								 {COMP_P_POSSING: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONSTRAINTS: [ARG_CONSTRAINT_OPTIONAL_POSSESSIVE]}}),
	"NOM-P-NP-ING": 			({COMP_ING_POC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_PP]},						COMP_PP: {ARG_CONSTRAINTS: [ARG_CONSTRAINT_REQUIRED_PREFIX]}},
								 {COMP_ING_POC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_PP]},						COMP_PP: {ARG_CONSTRAINTS: [ARG_CONSTRAINT_REQUIRED_PREFIX]}}),
	"NOM-POSSING-PP": 			({COMP_POSSING: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONSTRAINTS: [ARG_CONSTRAINT_OPTIONAL_POSSESSIVE]}},
								 {COMP_POSSING: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONSTRAINTS: [ARG_CONSTRAINT_OPTIONAL_POSSESSIVE]}}),
	"NOM-NP-P-POSSING": 		({COMP_P_POSSING: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONSTRAINTS: [ARG_CONSTRAINT_OPTIONAL_POSSESSIVE]}},
								 {COMP_P_POSSING: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONSTRAINTS: [ARG_CONSTRAINT_OPTIONAL_POSSESSIVE]}}),
	"NOM-NP-P-NP-ING": 			({COMP_ING_POC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_PP]},						COMP_PP: {ARG_CONSTRAINTS: [ARG_CONSTRAINT_REQUIRED_PREFIX]}},
								 {COMP_ING_POC: {ARG_ROOT_UPOSTAGS: [UPOS_VERB], ARG_ROOT_PATTERNS: [PATTERN_ING], ARG_CONTROLLED: [COMP_PP]},						COMP_PP: {ARG_CONSTRAINTS: [ARG_CONSTRAINT_REQUIRED_PREFIX]}}),

	# INF
	"NOM-FOR-TO-INF": 			({COMP_FOR_TO_INF: {ARG_ROOT_URELATIONS: [URELATION_TO, URELATION_NSUBJ], ARG_ROOT_UPOSTAGS: [UPOS_VERB, UPOS_AUX], ARG_CONSTRAINTS: [ARG_CONSTRAINT_REQUIRED_PREFIX]}},
								 {COMP_FOR_TO_INF: {ARG_ROOT_URELATIONS: [URELATION_TO, URELATION_NSUBJ], ARG_ROOT_UPOSTAGS: [UPOS_VERB, UPOS_AUX], ARG_CONSTRAINTS: [ARG_CONSTRAINT_REQUIRED_PREFIX]}}),
	"NOM-NP-TO-INF-OC": 		({COMP_TO_INF_OC: {ARG_ROOT_URELATIONS: [URELATION_TO], ARG_ROOT_UPOSTAGS: [UPOS_VERB, UPOS_AUX], ARG_CONTROLLED: [COMP_OBJ]}},
								 {COMP_TO_INF_OC: {ARG_ROOT_URELATIONS: [URELATION_TO], ARG_ROOT_UPOSTAGS: [UPOS_VERB, UPOS_AUX], ARG_CONTROLLED: [COMP_OBJ]}}),
	"NOM-NP-TO-INF-SC": 		({COMP_TO_INF_SC: {ARG_ROOT_URELATIONS: [URELATION_TO], ARG_ROOT_UPOSTAGS: [UPOS_VERB, UPOS_AUX], ARG_CONTROLLED: [COMP_SUBJ]}},
								 {COMP_TO_INF_SC: {ARG_ROOT_URELATIONS: [URELATION_TO], ARG_ROOT_UPOSTAGS: [UPOS_VERB, UPOS_AUX], ARG_CONTROLLED: [COMP_SUBJ]}}),
	"NOM-NP-TO-INF-VC": 		({COMP_TO_INF_VC: {ARG_ROOT_URELATIONS: [URELATION_TO], ARG_ROOT_UPOSTAGS: [UPOS_VERB, UPOS_AUX], ARG_CONTROLLED: [COMP_SUBJ, COMP_OBJ]}},
								 {COMP_TO_INF_VC: {ARG_ROOT_URELATIONS: [URELATION_TO], ARG_ROOT_UPOSTAGS: [UPOS_VERB, UPOS_AUX], ARG_CONTROLLED: [COMP_SUBJ, COMP_OBJ], ARG_CONTIGUOUS_TO: COMP_OBJ}}), # ONLY FOR NOM
	"NOM-TO-INF-SC":			({COMP_TO_INF_SC: {ARG_ROOT_URELATIONS: [URELATION_TO], ARG_ROOT_UPOSTAGS: [UPOS_VERB, UPOS_AUX], ARG_CONTROLLED: [COMP_SUBJ]}},
								 {COMP_TO_INF_SC: {ARG_ROOT_URELATIONS: [URELATION_TO], ARG_ROOT_UPOSTAGS: [UPOS_VERB, UPOS_AUX], ARG_CONTROLLED: [COMP_SUBJ]}}),
	"NOM-P-NP-TO-INF-OC":		({COMP_TO_INF_POC: {ARG_ROOT_URELATIONS: [URELATION_TO], ARG_ROOT_UPOSTAGS: [UPOS_VERB, UPOS_AUX], ARG_CONTROLLED: [COMP_PP]}},
								 {COMP_TO_INF_POC: {ARG_ROOT_URELATIONS: [URELATION_TO], ARG_ROOT_UPOSTAGS: [UPOS_VERB, UPOS_AUX], ARG_CONTROLLED: [COMP_PP]}}),
	"NOM-P-NP-TO-INF":			({COMP_TO_INF_POC: {ARG_ROOT_URELATIONS: [URELATION_TO], ARG_ROOT_UPOSTAGS: [UPOS_VERB, UPOS_AUX], ARG_CONTROLLED: [COMP_PP], ARG_CONTIGUOUS_TO: COMP_PP}},
								 {COMP_TO_INF_POC: {ARG_ROOT_URELATIONS: [URELATION_TO], ARG_ROOT_UPOSTAGS: [UPOS_VERB, UPOS_AUX], ARG_CONTROLLED: [COMP_PP], ARG_CONTIGUOUS_TO: COMP_PP}}),
	"NOM-P-NP-TO-INF-VC":		({COMP_TO_INF_VC: {ARG_ROOT_URELATIONS: [URELATION_TO], ARG_ROOT_UPOSTAGS: [UPOS_VERB, UPOS_AUX], ARG_CONTROLLED: [COMP_SUBJ, COMP_PP], ARG_CONTIGUOUS_TO: [COMP_PP]}},
								 {COMP_TO_INF_VC: {ARG_ROOT_URELATIONS: [URELATION_TO], ARG_ROOT_UPOSTAGS: [UPOS_VERB, UPOS_AUX], ARG_CONTROLLED: [COMP_SUBJ, COMP_PP], ARG_CONTIGUOUS_TO: [COMP_PP]}}),
	"NOM-PP-FOR-TO-INF":		({COMP_FOR_TO_INF: {ARG_ROOT_URELATIONS: [URELATION_TO, URELATION_NSUBJ], ARG_ROOT_UPOSTAGS: [UPOS_VERB, UPOS_AUX], ARG_CONSTRAINTS: [ARG_CONSTRAINT_REQUIRED_PREFIX]}},
								 {COMP_FOR_TO_INF: {ARG_ROOT_URELATIONS: [URELATION_TO, URELATION_NSUBJ], ARG_ROOT_UPOSTAGS: [UPOS_VERB, UPOS_AUX], ARG_CONSTRAINTS: [ARG_CONSTRAINT_REQUIRED_PREFIX]}}),
	"NOM-PP-HOW-TO-INF":		({COMP_HOW_TO_INF: {ARG_ROOT_URELATIONS: [URELATION_TO], ARG_ROOT_UPOSTAGS: [UPOS_VERB, UPOS_AUX]}},
								 {COMP_HOW_TO_INF: {ARG_ROOT_URELATIONS: [URELATION_TO], ARG_ROOT_UPOSTAGS: [UPOS_VERB, UPOS_AUX]}}),

	# SBAR
	"NOM-S":					({COMP_SBAR: {ARG_ROOT_URELATIONS: [URELATION_NSUBJ], ARG_ILLEGAL_PREFIXES: WHERE_WHEN_OPTIONS + HOW_OPTIONS + WH_VERB_OPTIONS}}, # The "that" complementizer word isn't required for verbs
								 {COMP_SBAR: {ARG_ROOT_URELATIONS: [URELATION_NSUBJ, URELATION_THAT]}}),
	"NOM-THAT-S":				({COMP_SBAR: {ARG_ROOT_URELATIONS: [URELATION_NSUBJ, URELATION_THAT]}},
								 {COMP_SBAR: {ARG_ROOT_URELATIONS: [URELATION_NSUBJ, URELATION_THAT]}}),
	"NOM-NP-S":					({COMP_SBAR: {ARG_ROOT_URELATIONS: [URELATION_NSUBJ], ARG_ILLEGAL_PREFIXES: WHERE_WHEN_OPTIONS + HOW_OPTIONS + WH_VERB_OPTIONS}}, # The "that" complementizer word isn't required for verbs
								 {COMP_SBAR: {ARG_ROOT_URELATIONS: [URELATION_NSUBJ, URELATION_THAT]}}),
	"NOM-PP-THAT-S":			({COMP_SBAR: {ARG_ROOT_URELATIONS: [URELATION_NSUBJ, URELATION_THAT]}},
								 {COMP_SBAR: {ARG_ROOT_URELATIONS: [URELATION_NSUBJ, URELATION_THAT]}}),

	# SUBJUNCT
	"NOM-S-SUBJUNCT":			({COMP_SBAR_SUBJUNCT: {ARG_CONSTRAINTS: [ARG_CONSTRAINT_SUBJUNCT], ARG_ROOT_URELATIONS: [URELATION_NSUBJ]}}, # The "that" complementizer word isn't required for verbs
								 {COMP_SBAR_SUBJUNCT: {ARG_CONSTRAINTS: [ARG_CONSTRAINT_SUBJUNCT], ARG_ROOT_URELATIONS: [URELATION_NSUBJ, URELATION_THAT]}}),
	"NOM-PP-THAT-S-SUBJUNCT":	({COMP_SBAR_SUBJUNCT: {ARG_CONSTRAINTS: [ARG_CONSTRAINT_SUBJUNCT], ARG_ROOT_URELATIONS: [URELATION_NSUBJ, URELATION_THAT]}},
								 {COMP_SBAR_SUBJUNCT: {ARG_CONSTRAINTS: [ARG_CONSTRAINT_SUBJUNCT], ARG_ROOT_URELATIONS: [URELATION_NSUBJ, URELATION_THAT]}}),
	"NOM-NP-AS-IF-S-SUBJUNCT":	({COMP_AS_IF_S_SUBJUNCT: {ARG_CONSTRAINTS: [ARG_CONSTRAINT_SUBJUNCT], ARG_ROOT_URELATIONS: [URELATION_NSUBJ]}},
								 {COMP_AS_IF_S_SUBJUNCT: {ARG_CONSTRAINTS: [ARG_CONSTRAINT_SUBJUNCT], ARG_ROOT_URELATIONS: [URELATION_NSUBJ]}})
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
	NOM_TYPE_SUBJ: [COMP_SUBJ],
	NOM_TYPE_OBJ: [COMP_OBJ],
	NOM_TYPE_IND_OBJ: [COMP_IND_OBJ, COMP_PP1, COMP_PP2, COMP_PP],

	# Ignored because they are rare and they don't refer to a specific complement (so they may have mistakes)
	# NOM_TYPE_P_OBJ: [COMP_PP],
	# NOM_TYPE_INSTRUMENT: [COMP_INSTRUMENT]
}