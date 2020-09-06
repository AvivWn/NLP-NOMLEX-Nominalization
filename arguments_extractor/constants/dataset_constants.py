from arguments_extractor.constants.lexicon_constants import *

# Tagsets labels
ARGS_TAGSET = {COMP_SUBJ:0, COMP_OBJ:1, COMP_IND_OBJ:2, COMP_PP:3, COMP_NONE:4}
ARGS_LABELS = list(ARGS_TAGSET.keys())

NOUNS_TAGSET = {NOM_TYPE_SUBJ:0, NOM_TYPE_OBJ:1, NOM_TYPE_IND_OBJ:2, NOM_TYPE_VERB_NOM:3, COMP_NONE:4}
NOUNS_LABELS = list(NOUNS_TAGSET.keys())