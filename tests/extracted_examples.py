EXTRACTIONS_BY_SENTENCE = {
	########################################################################
	# Predicates in passive-voice

	"The man was appointed by Apple.": {
		"appointed": [{"SUBJECT": "Apple", "OBJECT": "The man"}]},
	"The packages have been delivered by her.": {
		"delivered": [{"SUBJECT": "her", "OBJECT": "The packages"}]},
	"The document had been released into the public domain.": {
		"released": [{"PP": "into the public domain", "OBJECT": "The document"}]},
	"Bad advice was given.": {
		"advice": [],
		"given": [{"OBJECT": "Bad advice"}]},

	########################################################################
	# Complex NOM-TYPE

	# SUBJECT
	"They examined him.": {
		"examined": [{"SUBJECT": "They", "OBJECT": "him"}]},
	"The examiners of him.": {
		"examiners": [{"SUBJECT": "examiners", "OBJECT": "him"}]},

	# OBJECT
	"The man employed me.": {
		"employed": [{"SUBJECT": "The man", "OBJECT": "me"}]},
	"The man's employee.": {
		"employee": [{"SUBJECT": "The man", "OBJECT": "employee"}]},

	# IND-OBJECT
	"He consigned three paintings to Sotheby's.": {
		"consigned": [
			{"IND-OBJ": "Sotheby", "SUBJECT": "He", "OBJECT": "three paintings"},
			{"PP": "to Sotheby", "SUBJECT": "He", "OBJECT": "three paintings"}]},
	"The consignee of the three painting appreciated the gesture.": {
		"consignee": [{"IND-OBJ": "consignee", "OBJECT": "the three painting"}],
		"appreciated": [{"SUBJECT": "The consignee of the three painting", "OBJECT": "the gesture"}],
		"gesture": []},

	# P-OBJECT (IGNORED)
	"They captioned the picture with an humorous text.": {
		"captioned": [{"PP": "with an humorous text", "SUBJECT": "They", "OBJECT": "the picture"}]},
	"The caption of the picture was written in bold.": {
		"caption": [{"OBJECT": "the picture"}],
		"written": [{"OBJECT": "The caption of the picture"}]},

	# INSTRUMENT (IGNORED)
	"They lined the cubes in a row.": {
		"lined": [{"SUBJECT": "They", "OBJECT": "the cubes"}]},
	"The cube's liner in a row.": {
		"liner": [
			{"SUBJECT": "The cube"},
			{"OBJECT": "The cube"}]},

	########################################################################
	# Predicates with particles (PART)

	"They backed up on the disk.": {
		"backed": [{"PARTICLE": "up", "PP": "on the disk", "SUBJECT": "They"}]},
	"Their backup on the disk.": {
		"backup": [{"PP": "on the disk", "SUBJECT": "Their", "OBJECT": "backup"}]},
	"Their back-up on the disk.": {
		"back-up": [{"PP": "on the disk", "SUBJECT": "Their", "OBJECT": "back-up"}]},
	"They backed up the files to the cloud.": {
		"backed": [{"PARTICLE": "up", "PP": "to the cloud", "SUBJECT": "They", "OBJECT": "the files"}]},
	"Their backup of the files to the cloud.": {
		"backup": [
			{"OBJECT": "the files", "PP": "to the disk", "SUBJECT": "Their"},
			{"PP": "to the cloud", "SUBJECT": "Their", "OBJECT": "backup"},
			{"PP": "of the files", "SUBJECT": "Their", "OBJECT": "backup"}]},
	"Their back-up of the data to the disk.": {
		"back-up": [
			{"OBJECT": "the files", "PP": "to the disk", "SUBJECT": "Their"},
			{"PP": "to the cloud", "SUBJECT": "Their", "OBJECT": "back-up"},
			{"PP": "of the files", "SUBJECT": "Their", "OBJECT": "back-up"}]},

	########################################################################
	# Multi-word prepositions

	# 3-worded prepositions
	"They discussed whether he is sleeping.": {
		"discussed": [{"SBAR": "whether he is sleeping", "SUBJECT": "They"}],
		"sleeping": [{"SUBJECT": "he"}]},
	"Their discussion with respect to whether he is sleeping.": {
		"discussion": [{"SBAR": "whether he is sleeping", "SUBJECT": "Their"}],
		"respect": [],
		"sleeping": [{"SUBJECT": "he"}]},
	"Alice dreamed her boyfriend with regard to his future.": {
		"dreamed": [{"PP": "with regard to his future", "SUBJECT": "Alice", "OBJECT": "her boyfriend"}],
		"regard": []},
	"Alice's dream of her boyfriend with regard to his future.": {
		"dream": [
			{"SUBJECT": "her boyfriend", "PP": "with regard to his future", "OBJECT": "Alice"},
			{"OBJECT": "her boyfriend", "PP": "with regard to his future", "SUBJECT": "Alice"}],
		"regard": []},
	"Alice dreamed her boyfriend in connection with his future.": {
		"dreamed": [{"PP": "in connection with his future", "SUBJECT": "Alice", "OBJECT": "her boyfriend"}],
		"connection": [{"PP": "with his future"}]},
	"Alice's dream of her boyfriend in connection with his future.": {
		"dream": [
			{"SUBJECT": "her boyfriend", "PP": "in connection with his future", "OBJECT": "Alice"},
			{"OBJECT": "her boyfriend", "PP": "in connection with his future", "SUBJECT": "Alice"}],
		"connection": [{"PP": "with his future"}]},

	# 2-worded prepositions
	"She speculated with respect to whether the man will arrive.": {
		"speculated": [{"SBAR": "whether the man will arrive", "SUBJECT": "She"}],
		"respect": [],
		"will": [],
		"arrive": [{"SUBJECT": "the man"}]},
	"Her speculation with respect to whether the man will arrive.": {
		"speculation": [{"SBAR": "whether the man will arrive", "SUBJECT": "Her"}],
		"respect": [],
		"will": [],
		"arrive": [{"SUBJECT": "the man"}]},
	"He shoot away from the target last night.": {
		"shoot": [
			{"PP": "away from the target", "SUBJECT": "He"},
			{"PP": "away from the target", "OBJECT": "He"}],
		"target": []},
	"His shoot away from the target last night.": {
		"shoot": [{"PP": "away from the target", "SUBJECT": "His"}],
		"target": []},
	"He started the game ahead of the others.": {
		"started": [{"PP": "ahead of the others", "SUBJECT": "He", "OBJECT": "the game"}]},
	"His start of the game ahead of the others.": {
		"start": [{"PP": "ahead of the others", "SUBJECT": "His", "OBJECT": "the game"}]},
	"He stationed the soldier next to his house.": {
		"stationed": [{"PP": "next to his house", "SUBJECT": "He", "OBJECT": "the soldier"}]},
	"The soldier's station by him next to his house.": {
		"station": [{"PP": "next to his house", "OBJECT": "The soldier", "SUBJECT": "him"}]},
	"She speculated as to whether the man will arrive.": {
		"speculated": [{"SBAR": "whether the man will arrive", "SUBJECT": "She"}],
		"will": [],
		"arrive": [{"SUBJECT": "the man"}]},
	"Her speculation as to whether the man will arrive.": {
		"speculation": [{"SBAR": "whether the man will arrive", "SUBJECT": "Her"}],
		"will": [],
		"arrive": [{"SUBJECT": "the man"}]},

	########################################################################
	# Complex constraints

	# DET-POSS-ONLY
	"He destroyed Paris.": {
		"destroyed": [{"SUBJECT": "He", "OBJECT": "Paris"}]},
	"Paris's destruction.": {
		"destruction": [{"OBJECT": "Paris"}]},

	# N-N-MOD-ONLY
	"He registered the child.": {
		"registered": [{"SUBJECT": "He", "OBJECT": "the child"}]},
	"The child registration.": {
		"registration": [{"OBJECT": "child"}]},

	# NOT constraints
	"The man's reflection with regard to the sun.": {
		"reflection": [{"PP": "with regard to the sun", "SUBJECT": "The man"}],
		"regard": []},
	"The man reflection with regard to the sun.": {
		"reflection": [{"PP": "with regard to the sun"}],
		"regard": []},
	"His representation of the paper to the audience.": {
		"representation": [{"PP": "to the audience", "SUBJECT": "His", "OBJECT": "the paper"}]},
	"The paper's representation by them to the audience.": {
		"representation": [{"OBJECT": "The paper", "SUBJECT": "them"}]},
	"The paper man representation.": {
		"representation": [
			{"SUBJECT": "man"},
			{"SUBJECT": "paper"},
			{"OBJECT": "paper"},
			{"OBJECT": "man"}]},

	# SUBJ-OBJ-ALT (SUBJECT is actually OBJECT)
	# standard
	"Microsoft circulated the rumor.": {
		"circulated": [{"SUBJECT": "Microsoft", "OBJECT": "the rumor"}],
		"rumor": []},
	"Microsoft's circulation of the rumor.": {
		"circulation": [
			{"SUBJECT": "Microsoft", "OBJECT": "the rumor"},
			{"OBJECT": "Microsoft", "SUBJECT": "the rumor"}],
		"rumor": []},
	# ALTERNATES
	"The rumor circulated.": {
		"rumor": [],
		"circulated": [{"OBJECT": "The rumor"}]},
	"The circulation of the rumor.": {
		"circulation": [{"OBJECT": "the rumor"}],
		"rumor": []},

	# SUBJ-IND-OBJ-ALT (SUBJECT is actually IND-OJB)
	# standard
	"The man rented Bill a boat.": {
		"rented": [{"IND-OBJ": "Bill", "SUBJECT": "The man", "OBJECT": "a boat"}]},
	"The man's rental of a boat, to Bill.": {
		"rental": [{"IND-OBJ": "Bill", "SUBJECT": "The man", "OBJECT": "a boat"}]},
	# ALTERNATES-OPT
	"Bill rented a boat.": {
		"rented": [
			{"SUBJECT": "Bill", "OBJECT": "a boat"},
			{"IND-OBJ": "Bill", "OBJECT": "a boat"}]},
	"Bill's rental of a boat.": {
		"rental": [
			{"IND-OBJ": "Bill", "OBJECT": "a boat"},
			{"SUBJECT": "Bill", "OBJECT": "a boat"}]},

	########################################################################
	# WH-S complements

	# NOM-PP-HOW-TO-INF
	"She described how to do it.": {
		"described": [{"SBAR": "how to do it", "SUBJECT": "She"}],
		"do": [{"OBJECT": "it"}],
	},
	"Her description of how to do it.": {
		"description": [{"SBAR": "how to do it", "SUBJECT": "Her"}],
		"do": [{"OBJECT": "it"}],
	},
	"The president explained to the man how to cheat his wife": {
		"explained": [{"PP": "to the man", "SBAR": "how to cheat his wife", "SUBJECT": "The president"}],
		"cheat": [{"OBJECT": "his wife"}]},
	"The president's explanation to the man, of how to cheat his wife": {
		"explanation": [{"PP": "to the man,", "SBAR": "how to cheat his wife", "SUBJECT": "The president"}],
		"cheat": [{"OBJECT": "his wife"}]},

	# NOM-WHERE-WHEN-S
	"He knows when the blizzard would come.": {
		"knows": [{"SBAR": "when the blizzard would come", "SUBJECT": "He"}],
		"come": [{"SUBJECT": "the blizzard"}]},
	"His knowledge of when the blizzard would come.": {
		"knowledge": [{"SBAR": "when the blizzard would come", "SUBJECT": "His"}],
		"come": [{"SUBJECT": "the blizzard"}]},
	"He predicted where the blizzard would strike.": {
		"predicted": [{"SBAR": "where the blizzard would strike", "SUBJECT": "He"}],
		"strike": [{"OBJECT": "the blizzard"}]},
	"The predictor of where the blizzard would strike.": {
		"predictor": [{"SBAR": "where the blizzard would strike", "SUBJECT": "predictor"}],
		"strike": [{"OBJECT": "the blizzard"}]},
	"The man forecasts how many people will arrive.": {
		"forecasts": [{"SBAR": "how many people will arrive", "SUBJECT": "The man"}],
		"will": [],
		"arrive": [{"SUBJECT": "how many people"}]},
	"His forecast of how many people will arrive.": {
		"forecast": [{"SBAR": "how many people will arrive", "SUBJECT": "His"}],
		"will": [],
		"arrive": [{"SUBJECT": "how many people"}]},
	"They forecasted how many will be there.": {
		"forecasted": [{"SBAR": "how many will be there", "SUBJECT": "They"}]},
	"Their forecast of how many will be there.": {
		"forecast": [{"SBAR": "how many will be there", "SUBJECT": "Their"}]},
	"She forecasts how much money they will make.": {
		"forecasts": [{"SBAR": "how much money they will make", "SUBJECT": "She"}], "will": [],
		"make": [{"SUBJECT": "they", "OBJECT": "how much money"}]},
	"Her forecast of how much money they will make.": {
		"forecast": [{"SBAR": "how much money they will make", "SUBJECT": "Her"}],
		"will": [],
		"make": [{"SUBJECT": "they", "OBJECT": "how much money"}]},

	# NOM-NP-P-WH-S
	"I have pressed him about whether we would meet.": {
		"pressed": [{"SBAR": "whether we would meet", "SUBJECT": "I", "OBJECT": "him"}],
		"meet": [{"SUBJECT": "we"}]},
	"The pressure of him by me, regarding whether we would meet.": {
		"pressure": [{"SBAR": "whether we would meet", "SUBJECT": "me", "OBJECT": "him"}],
		"meet": [{"SUBJECT": "we"}]},
	"I pressed them about what they want.": {
		"pressed": [{"SBAR": "what they want", "SUBJECT": "I", "OBJECT": "them"}]},
	"The pressure of them by me about what they want.": {
		"pressure": [{"SBAR": "what they want", "SUBJECT": "me", "OBJECT": "them"}]},
	"The man pressed the woman, with respect to whether he should kill them.": {
		"pressed": [{"SBAR": "whether he should kill them", "SUBJECT": "The man", "OBJECT": "the woman"}],
		"respect": [],
		"kill": [{"SUBJECT": "he", "OBJECT": "them"}]},
	"The woman's pressure by the man, with respect to whether he should kill them.": {
		"pressure": [{"SBAR": "whether he should kill them", "OBJECT": "The woman", "SUBJECT": "the man"}],
		"respect": [],
		"kill": [{"SUBJECT": "he", "OBJECT": "them"}]},

	# NOM-PP-P-WH-S
	"I argued with him about whether he should kill them.": {
		"argued": [{"SBAR": "whether he should kill them", "PP": "with him", "SUBJECT": "I"}],
		"kill": [{"SUBJECT": "he", "OBJECT": "them"}]},
	"My argument with him about whether he should kill them.": {
		"argument": [{"SBAR": "whether he should kill them", "PP": "with him", "SUBJECT": "My"}],
		"kill": [{"SUBJECT": "he", "OBJECT": "them"}]},
	"I argued with them about what they needed to do.": {
		"argued": [{"SBAR": "what they needed to do", "PP": "with them", "SUBJECT": "I"}],
		"needed": [{"TO-INF": "to do", "SUBJECT": "they"}]},
	"My argument with them about what they needed to do.": {
		"argument": [{"SBAR": "what they needed to do", "PP": "with them", "SUBJECT": "My"}],
		"needed": [{"TO-INF": "to do", "SUBJECT": "they"}]},
	"I argued with him about whether to kill her.": {
		"argued": [{"SBAR": "whether to kill her", "PP": "with him", "SUBJECT": "I"}],
		"kill": [{"OBJECT": "her"}]},
	"My argument with him about whether to kill her.": {
		"argument": [{"SBAR": "whether to kill her", "PP": "with him", "SUBJECT": "My"}],
		"kill": [{"OBJECT": "her"}]},
	"I argued with them about what to do.": {
		"argued": [{"SBAR": "what to do", "PP": "with them", "SUBJECT": "I"}]},
	"My argument with them about what to do.": {
		"argument": [{"SBAR": "what to do", "PP": "with them", "SUBJECT": "My"}]},

	# NOM-P-WH-S
	"He inquired about what to do.": {
		"inquired": [{"SBAR": "what to do", "SUBJECT": "He"}]},
	"His inquiry about what to do.": {
		"inquiry": [{"SBAR": "what to do", "SUBJECT": "His"}]},
	"John inquired about whether or not they should attend": {
		"inquired": [{"SBAR": "whether or not they should attend", "SUBJECT": "John"}],
		"attend": [{"SUBJECT": "they"}]},
	"John's inquiry concerning whether or not they should attend": {
		"inquiry": [{"SBAR": "whether or not they should attend", "SUBJECT": "John"}],
		"attend": [{"SUBJECT": "they"}]},

	# NOM-PP-WH-S
	"He explained to us what to do next.": {
		"explained": [{"SBAR": "what to do next", "PP": "to us", "SUBJECT": "He"}],
		"do": [{"MODIFIER": "next"}]},
	"His explanation to us of what to do next.": {
		"explanation": [{"SBAR": "what to do next", "PP": "to us", "SUBJECT": "His"}],
		"do": [{"MODIFIER": "next"}]},

	# NOM-NP-WH-S
	"He reminded me whether the world is round.": {
		"reminded": [{"SBAR": "whether the world is round", "SUBJECT": "He", "OBJECT": "me"}]},
	"His reminder to me of whether the world is round.": {
		"reminder": [{"SBAR": "whether the world is round", "SUBJECT": "reminder", "OBJECT": "me"}]},

	# NOM-HOW-S
	"They know how it was done.": {
		"know": [{"SUBJECT": "They", "SBAR": "how it was done"}],
		"done": [{"OBJECT": "it"}]},
	"Their knowledge of how it was done.": {
		"knowledge": [{"SUBJECT": "Their", "SBAR": "how it was done"}],
		"done": [{"OBJECT": "it"}]},
	"They analyzed how the machine works.": {
		"analyzed": [{"SUBJECT": "They", "SBAR": "how the machine works"}],
		"works": [{"SUBJECT": "the machine"}]},
	"The analysis by the man, of how they played.": {
		"analysis": [{"SUBJECT": "the man,", "SBAR": "how they played"}],
		"played": [{"SUBJECT": "they"}]},

	# NOM-WH-S
	"I wonder whether he is referring to Jake.": {
		"wonder": [{"SBAR": "whether he is referring to Jake", "SUBJECT": "I"}],
		"referring": [{"PP": "to Jake", "SUBJECT": "he"}]},
	"My wonder of whether he is referring to Jake.": {
		"wonder": [{"SBAR": "whether he is referring to Jake", "SUBJECT": "My"}],
		"referring": [{"PP": "to Jake", "SUBJECT": "he"}]},
	"I know what he will do.": {
		"know": [{"SBAR": "what he will do", "SUBJECT": "I"}],
		"will": []},
	"My knowledge of what he will do.": {
		"knowledge": [{"SBAR": "what he will do", "SUBJECT": "My"}],
		"will": []},
	"I wonder if he is sick.": {
		"wonder": [{"SBAR": "if he is sick", "SUBJECT": "I"}]},
	"My wonder of whether he is sick.": {
		"wonder": [{"SBAR": "whether he is sick", "SUBJECT": "My"}]},

	########################################################################
	# TO-INF complements

	# NOM-PP-TO-INF-RECIP (IGNORED)
	"Henry conspired with Marsha to steal the contract.": {
		"conspired": [
			{"PP": "with Marsha", "SUBJECT": "Henry"},
			{"TO-INF": "to steal the contract", "SUBJECT": "Henry"}],
		"contract": []},
	"Henry's conspiracy with Marsha to steal the contract.": {
		"conspiracy": [
			{"PP": "with Marsha", "SUBJECT": "Henry"},
			{"TO-INF": "to steal the contract", "SUBJECT": "Henry"}],
		"contract": []},
	"Henry and Tomer conspired to steal the contract.": {
		"conspired": [{"TO-INF": "to steal the contract", "SUBJECT": "Henry and Tomer"}],
		"contract": []},
	"Henry and Tomer's conspiracy to steal the contract.": {
		"conspiracy": [{"TO-INF": "to steal the contract", "SUBJECT": "Henry and Tomer's"}],
		"contract": []},
	"The Henry and Tomer conspiracy to steal the contract.": {
		"conspiracy": [{"TO-INF": "to steal the contract", "SUBJECT": "Tomer"}],  # not quite
		"contract": []},
	"Their conspiracy to steal the contract.": {
		"conspiracy": [{"TO-INF": "to steal the contract", "SUBJECT": "Their"}],
		"contract": []},

	# NOM-PP-FOR-TO-INF
	"Daniel arranged with her for John to take the bus to school.": {
		"arranged": [{"TO-INF": "John to take the bus to school", "PP": "with her", "SUBJECT": "Daniel"}],
		"take": [
			{"PP": "to school", "SUBJECT": "John", "OBJECT": "the bus"},
			{"IND-OBJ": "school", "SUBJECT": "John", "OBJECT": "the bus"}]},
	"The arrangement by Daniel with her for John to take the bus to school.": {
		"arrangement": [
			{"TO-INF": "John to take the bus to school", "PP": "with her", "SUBJECT": "Daniel"}],
		"take": [
			{"PP": "to school", "SUBJECT": "John", "OBJECT": "the bus"},
			{"IND-OBJ": "school", "SUBJECT": "John", "OBJECT": "the bus"}]},
	"Daniel arranged with her for John to be the main actor.": {
		"arranged": [{"TO-INF": "John to be the main actor", "PP": "with her", "SUBJECT": "Daniel"}],
		"actor": [{"MODIFIER": "main", "SUBJECT": "actor"}]},
	"Daniel's arrangement with her for John to be the main actor.": {
		"arrangement": [{"TO-INF": "John to be the main actor", "PP": "with her", "SUBJECT": "Daniel"}],
		"actor": [{"MODIFIER": "main", "SUBJECT": "actor"}]},

	# NOM-FOR-TO-INF
	"Sue calls for John to tell him that he forgot his keys.": {
		"calls": [{"TO-INF": "John to tell him that he forgot his keys", "SUBJECT": "Sue"}]},
	"Sue's call for John to tell him that he forgot his keys.": {
		"call": [{"TO-INF": "John to tell him that he forgot his keys", "SUBJECT": "Sue"}]},

	# NOM-P-NP-TO-INF
	"He relies on her to succeed.": {
		"relies": [{"PP": "on her", "SUBJECT": "He", "TO-INF": "to succeed"}],
		"succeed": []},
	"His reliance on her to succeed, didn't help.": {
		"reliance": [{"PP": "on her", "SUBJECT": "His", "TO-INF": "to succeed"}],
		"help": [{"SUBJECT": "His reliance on her to succeed"}]},
	"He relies on her to be successful.": {
		"relies": [{"PP": "on her", "SUBJECT": "He", "TO-INF": "to be successful"}]},
	"His reliance on her to be successful, didn't help.": {
		"reliance": [{"PP": "on her", "SUBJECT": "His", "TO-INF": "to be successful"}],
		"help": [{"SUBJECT": "His reliance on her to be successful"}]},

	# NOM-P-NP-TO-INF-VC
	"She appealed to him to leave the compound.": {
		"appealed": [{"PP": "to him", "TO-INF": "to leave the compound", "SUBJECT": "She"}],
		"leave": [{"OBJECT": "the compound"}]},
	"Her appeal to him to leave the compound.": {
		"appeal": [{"PP": "to him", "TO-INF": "to leave the compound", "SUBJECT": "Her"}],
		"leave": [{"OBJECT": "the compound"}]},

	# NOM-P-NP-TO-INF-OC
	"I imposed on him to go to school.": {
		"imposed": [{"PP": "on him", "SUBJECT": "I", "TO-INF": "to go to school"}],
		"go": [{"PP": "to school"}]},
	"My imposition on him to go to school changed his life.": {
		"imposition": [{"PP": "on him", "SUBJECT": "My", "TO-INF": "to go to school"}],
		"life": [
			{"OBJECT": "his"},
			{"SUBJECT": "his"}],
		"go": [{"PP": "to school"}]},

	# NOM-NP-TO-INF-VC
	"She solicited Mayor Koch to lead the parade.": {
		"solicited": [{"TO-INF": "to lead the parade", "SUBJECT": "She", "OBJECT": "Mayor Koch"}],
		"lead": [{"OBJECT": "the parade"}]}, "Her solicitation of Mayor Koch to lead the parade.": {
		"solicitation": [{"TO-INF": "to lead the parade", "SUBJECT": "Her", "OBJECT": "Mayor Koch"}],
		"lead": [{"OBJECT": "the parade"}]},

	# NOM-NP-TO-INF-SC
	"John promised Mary to fix the desk lamp.": {
		"promised": [{"TO-INF": "to fix the desk lamp", "SUBJECT": "John", "OBJECT": "Mary"}],
		"fix": [{"OBJECT": "the desk lamp"}]},
	"John's promise to Mary to fix the desk lamp.": {
		"promise": [{"TO-INF": "to fix the desk lamp", "SUBJECT": "John", "OBJECT": "Mary"}],
		"fix": [{"OBJECT": "the desk lamp"}]},

	# NOM-NP-TO-INF-OC
	"We designated Allie to drive us home from the party.": {
		"designated": [{"SUBJECT": "We", "OBJECT": "Allie", "TO-INF": "to drive us home from the party"}],
		"drive": [{"PP": "from the party", "OBJECT": "us"}],
		"party": [{"SUBJECT": "party"}]},
	"Our designation of Allie to drive us home from the party.": {
		"designation": [{"SUBJECT": "Our", "OBJECT": "Allie", "TO-INF": "to drive us home from the party"}],
		"drive": [{"PP": "from the party", "OBJECT": "us"}],
		"party": [{"SUBJECT": "party"}]},

	# NOM-TO-INF-SC
	"He needs to win every argument.": {
		"needs": [{"TO-INF": "to win every argument", "SUBJECT": "He"}],
		"win": [{"OBJECT": "every argument"}],
		"argument": []},
	"His need to win every argument.": {
		"need": [{"TO-INF": "to win every argument", "SUBJECT": "His"}],
		"win": [{"OBJECT": "every argument"}],
		"argument": []},

	########################################################################
	# POSSING complements

	# NOM-POSSING-PP
	"He discusses with the crowd their giving up smoking.": {
		"discusses": [{"SUBJECT": "He", "PP": "with the crowd", "ING": "their giving up smoking"}],
		"giving": [{"SUBJECT": "their", "PARTICLE": "up", "OBJECT": "smoking"}]},
	"His discussion with the crowd about their giving up smoking.": {
		"discussion": [{"SUBJECT": "His", "PP": "with the crowd", "ING": "their giving up smoking"}],
		"giving": [{"SUBJECT": "their", "PARTICLE": "up", "OBJECT": "smoking"}]},
	# "He discusses their filming, with them.": {
	# 	"discusses": [{"SUBJECT": "He", "ING": "their filming", "PP": "with them"}]},
	# "His discussion with them about their filming the movie.": {
	# 	"discussion": [{"SUBJECT": "His", "PP": "with them", "ING": "about filming the movie"}]},
	"Clare explained to me planning to steal the bank.": {
		"explained": [{"SUBJECT": "Clare", "PP": "to me", "ING": "planning to steal the bank"}],
		"planning": [{"TO-INF": "to steal the bank"}]},
	"Clare's explanation to me of planning to steal the bank.": {
		"explanation": [{"SUBJECT": "Clare", "PP": "to me", "ING": "planning to steal the bank"}],
		"planning": [{"TO-INF": "to steal the bank"}]},
	
	# "He recommended to  their giving money to poverty.": {
	# 	"recommended": [{"PP": "to the children", "SUBJECT": "He", "ING": "their giving money to poverty"}],
	# 	"giving": [{"SUBJECT": "their", "OBJECT": "money", "IND-OBJ": "poverty"}]},
	# "His recommendation to the children, of their giving money to poverty.": {
	# 	"recommendation": [{"PP": "to the children,", "SUBJECT": "His", "ING": "their giving money to poverty"}],
	# 	"giving": [{"SUBJECT": "their", "OBJECT": "money", "IND-OBJ": "poverty"}]},
	# "He emphasized working hard at night, to the children."
	# "He  working hard at night, to the children."
	# "His emphasis to the children, of working hard at night."
	# "His recommendation of giving up learning in school, to the children.": {
	# 	"recommendation": [{"PP": "to the children", "SUBJECT": "Their", "ING": "their winning the war"}],
	# 	"winning": [{"SUBJECT": "their", "OBJECT": "the war"}],
	# 	"war": []},
	# "They recommended winning the war, to the soldiers.": {
	# 	"recommended": [{"PP": "to the soldiers", "SUBJECT": "They", "ING": "winning the war"}],
	# 	"winning": [{"OBJECT": "the war"}],
	# 	"war": []},
	# "Their recommendation of winning the war, to the soldiers.": {
	# 	"recommendation": [{"PP": "to the soldiers", "SUBJECT": "Their", "ING": "winning the war"}],
	# 	"winning": [{"OBJECT": "the war"}],
	# 	"war": []},

	# NOM-PP-P-POSSING
	"Jake disagreed with Mick, about Clinton's visiting to China.": {
		"disagreed": [{"PP": "with Mick,", "SUBJECT": "Jake", "ING": "Clinton's visiting to China"}],
		"visiting": [{"SUBJECT": "Clinton"}]},
	"The disagreement by Jake with Mick, about Clinton's visiting to China.": {
		"disagreement": [{"PP": "with Mick,", "SUBJECT": "Jake", "ING": "Clinton's visiting to China"}],
		"visiting": [{"SUBJECT": "Clinton"}]},
	"Jake disagreed with Mick about his visiting to China.": {
		"disagreed": [{"PP": "with Mick", "SUBJECT": "Jake", "ING": "his visiting to China"}],
		"visiting": [{"SUBJECT": "his"}]},
	"Jake's disagreement with Mick about his visiting to China.": {
		"disagreement": [{"PP": "with Mick", "SUBJECT": "Jake", "ING": "his visiting to China"}],
		"visiting": [{"SUBJECT": "his"}]},
	"Jake disagreed with Mick about visiting to China.": {
		"disagreed": [{"PP": "with Mick", "SUBJECT": "Jake", "ING": "visiting to China"}],
		"visiting": []},
	"Jake's disagreement with Mick about visiting to China.": {
		"disagreement": [{"PP": "with Mick", "SUBJECT": "Jake", "ING": "visiting to China"}],
		"visiting": []},

	# NOM-NP-P-POSSING
	"We collected money for John's sweeping the road.": {
		"collected": [{"SUBJECT": "We", "OBJECT": "money", "ING": "John's sweeping the road"}],
		"sweeping": [{"SUBJECT": "John", "OBJECT": "the road"}]},
	"Our collection of money for John's sweeping the road, helped him.": {
		"collection": [{"SUBJECT": "Our", "OBJECT": "money", "ING": "John's sweeping the road"}],
		"sweeping": [{"SUBJECT": "John", "OBJECT": "the road"}],
		"helped": [{"SUBJECT": "Our collection of money for John's sweeping the road,", "OBJECT": "him"}]},
	"We collected money for her sweeping the road.": {
		"collected": [{"SUBJECT": "We", "OBJECT": "money", "ING": "her sweeping the road"}],
		"sweeping": [{"SUBJECT": "her", "OBJECT": "the road"}]},
	"Our collection of money for her sweeping the road.": {
		"collection": [{"SUBJECT": "Our", "OBJECT": "money", "ING": "her sweeping the road"}],
		"sweeping": [{"SUBJECT": "her", "OBJECT": "the road"}]},
	"We collected money for sweeping the road.": {
		"collected": [{"SUBJECT": "We", "OBJECT": "money", "ING": "sweeping the road"}],
		"sweeping": [{"OBJECT": "the road"}]},
	"Our collection of money for sweeping the road.": {
		"collection": [{"SUBJECT": "Our", "OBJECT": "money", "ING": "sweeping the road"}],
		"sweeping": [{"OBJECT": "the road"}]},

	# NOM-P-POSSING
	"Jay argued against John's stealing the money.": {
		"argued": [{"SUBJECT": "Jay", "ING": "John's stealing the money"}]},
	"The argument by Jay against John's stealing the money.": {
		"argument": [{"SUBJECT": "Jay", "ING": "John's stealing the money"}]},
	"Jay argued against his drinking the wine last night.": {
		"argued": [{"SUBJECT": "Jay", "ING": "his drinking the wine last night"}],
		"drinking": [{"SUBJECT": "his", "OBJECT": "the wine"}]},
	"Jay's argument against his drinking the wine last night.": {
		"argument": [{"SUBJECT": "Jay", "ING": "his drinking the wine last night"}],
		"drinking": [{"SUBJECT": "his", "OBJECT": "the wine"}]},
	"Jay argued against paying for the damage.": {
		"argued": [{"SUBJECT": "Jay", "ING": "paying for the damage"}],
		"paying": [{"PP": "for the damage"}],
		"damage": []},
	"Jay's argument against paying for the damage.": {
		"argument": [{"SUBJECT": "Jay", "ING": "paying for the damage"}],
		"paying": [{"PP": "for the damage"}],
		"damage": []},

	# NOM-POSSING
	"My father suggested his taking out loans to pay for college.": {
		"suggested": [{"SUBJECT": "My father", "ING": "his taking out loans to pay for college"}],
		"taking": [{"PARTICLE": "out", "SUBJECT": "his", "OBJECT": "loans to pay for college"}],
		"loans": [],
		"pay": [{"PP": "for college"}]},
	"My father's suggestion of his taking out loans to pay for college.": {
		"suggestion": [{"SUBJECT": "My father", "ING": "his taking out loans to pay for college"}],
		"taking": [{"PARTICLE": "out", "SUBJECT": "his", "OBJECT": "loans to pay for college"}],
		"loans": [],
		"pay": [{"PP": "for college"}]},
	"Tom suggested Aviv's taking out loans to pay for college.": {
		"suggested": [{"SUBJECT": "Tom", "ING": "Aviv's taking out loans to pay for college"}],
		"taking": [{"PARTICLE": "out", "SUBJECT": "Aviv", "OBJECT": "loans to pay for college"}],
		"loans": [],
		"pay": [{"PP": "for college"}]},
	"The suggestion by Tom, of Aviv's taking out loans to pay for college.": {
		"suggestion": [{"SUBJECT": "Tom,", "ING": "Aviv's taking out loans to pay for college"}],
		"taking": [{"PARTICLE": "out", "SUBJECT": "Aviv", "OBJECT": "loans to pay for college"}],
		"loans": [],
		"pay": [{"PP": "for college"}]},
	"The administration abolished sitting in on classes.": {
		"administration": [],
		"abolished": [{"SUBJECT": "The administration", "ING": "sitting in on classes"}],
		"sitting": [{"PP": "on classes", "PARTICLE": "in"}]},
	"The administration's abolition of sitting in on classes.": {
		"administration": [],
		"abolition": [{"SUBJECT": "The administration", "ING": "sitting in on classes"}],
		"sitting": [{"PP": "on classes", "PARTICLE": "in"}]},

	########################################################################
	# ING complements

	# NOM-NP-P-NP-ING
	"I called him about them, leaving the country.": {
		"called": [{"ING": "leaving the country", "PP": "about them", "OBJECT": "him", "SUBJECT": "I"}],
		"leaving": [{"OBJECT": "the country"}]},
	"My call for him about them leaving the country.": {
		"call": [{"ING": "leaving the country", "PP": "about them", "OBJECT": "him", "SUBJECT": "My"}],
		"leaving": [{"OBJECT": "the country", "SUBJECT": "about them"}]},
	"I educated him about Joe, being a man.": {
		"educated": [{"ING": "being a man", "PP": "about Joe", "OBJECT": "him", "SUBJECT": "I"}]},

	# NOM-P-NP-ING
	"I thought about the student, failing the test.": {
		"thought": [{"ING": "failing the test", "PP": "about the student", "SUBJECT": "I"}],
		"student": [{"SUBJECT": "student"}],
		"failing": [{"OBJECT": "the test"}],
		"test": []},
	"My thinking about the student, chasing after his friend.": {
		"thinking": [{"ING": "chasing after his friend", "PP": "about the student", "SUBJECT": "My"}],
		"student": [{"SUBJECT": "student"}],
		"chasing": [{"PP": "after his friend"}]},

	# NOM-NP-P-ING-SC
	"They justified their delay by riding the bikes.": {
		"justified": [{"ING": "riding the bikes", "SUBJECT": "They", "OBJECT": "their delay"}],
		"delay": [
			{"SUBJECT": "their"},
			{"OBJECT": "their"}],
		"riding": [{"OBJECT": "the bikes"}]},
	"Their justification of their delay by riding the bikes.": {
		"justification": [{"ING": "riding the bikes", "SUBJECT": "Their", "OBJECT": "their delay"}],
		"delay": [
			{"SUBJECT": "their"},
			{"OBJECT": "their"}],
		"riding": [{"OBJECT": "the bikes"}]},

	# NOM-NP-P-ING-OC
	"The state imprisoned the congressman for failing to pay taxes.": {
		"imprisoned": [{"SUBJECT": "The state", "OBJECT": "the congressman", "ING": "failing to pay taxes"}],
		"failing": [{"TO-INF": "to pay taxes"}], "pay": [{"OBJECT": "taxes"}], "taxes": []},
	"The state's imprisonment of the congressman for failing to pay taxes.": {
		"imprisonment": [{"SUBJECT": "The state", "OBJECT": "the congressman", "ING": "failing to pay taxes"}],
		"failing": [{"TO-INF": "to pay taxes"}],
		"pay": [{"OBJECT": "taxes"}],
		"taxes": []},

	# NOM-NP-P-ING
	"I prohibited the man from drinking wine.": {
		"prohibited": [{"SUBJECT": "I", "NP": "the man", "ING": "drinking wine"}],
		"drinking": [{"OBJECT": "wine"}]},
	"My prohibition of the man from drinking wine.": {
		"prohibition": [{"SUBJECT": "My", "NP": "the man", "ING": "drinking wine"}],
		"drinking": [{"OBJECT": "wine"}]},

	# NOM-P-ING-SC
	"They confessed to lying at poker.": {
		"confessed": [{"ING": "lying at poker", "SUBJECT": "They"}],
		"poker": [{"SUBJECT": "poker"}]},
	"Their confession to lying at poker.": {
		"confession": [{"ING": "lying at poker", "SUBJECT": "Their"}],
		"poker": [{"SUBJECT": "poker"}]},
	"They confessed to their stealing the store.": {
		"confessed": [{"ING": "their stealing the store", "SUBJECT": "They"}]},
	"Their confession to their stealing the store.": {
		"confession": [{"ING": "their stealing the store", "SUBJECT": "Their"}]},

	# NOM-NP-ING-OC
	"I imitated the president, denying the charges.": {
		"imitated": [{"ING": "denying the charges", "OBJECT": "the president", "SUBJECT": "I"}],
		"denying": [{"OBJECT": "the charges"}],
		"charges": []},
	"My imitation of the president, denying the charges.": {
		"imitation": [{"ING": "denying the charges", "OBJECT": "the president", "SUBJECT": "My"}],
		"denying": [{"OBJECT": "the charges"}],
		"charges": []},

	# NOM-NP-ING-SC
	"He risked his life chasing after her.": {
		"risked": [{"ING": "chasing after her", "SUBJECT": "He", "OBJECT": "his life"}],
		"life": [
			{"SUBJECT": "his"},
			{"OBJECT": "his"}],
		"chasing": [{"PP": "after her"}]},
	"His life's risk by him chasing after her, made me proud.": {
		"life": [
			{"SUBJECT": "His"},
			{"OBJECT": "His"}],
		"risk": [{"ING": "chasing after her", "OBJECT": "His life", "SUBJECT": "him"}],
		"chasing": [{"PP": "after her"}],
		"made": []},

	# NOM-NP-ING
	"I justified them cheating in the test.": {
		"justified": [{"ING": "cheating in the test", "NP": "them", "SUBJECT": "I"}],
		"cheating": [],
		"test": []},
	"The justification of them cheating the test, proved the point.": {
		"justification": [{"ING": "cheating the test", "NP": "them"}],
		"cheating": [{"OBJECT": "the test"}],
		"proved": [{"SUBJECT": "The justification of them cheating the test", "OBJECT": "the point"}]},

	# NOM-ING-SC
	"The police department continued accepting bribes.": {
		"continued": [{"ING": "accepting bribes", "SUBJECT": "The police department"}],
		"accepting": [{"OBJECT": "bribes"}],
		"bribes": []},
	"The police department's continuance of accepting bribes.": {
		"continuance": [{"ING": "accepting bribes", "SUBJECT": "The police department"}],
		"accepting": [{"OBJECT": "bribes"}],
		"bribes": []},

	########################################################################
	# S complements

	# NOM-NP-AS-IF-S-SUBJUNCT (SUBJUNCT is ignored)
	"They regarded him as if he were an animal.": {
		"regarded": [{"SBAR": "as if he were an animal", "SUBJECT": "They", "OBJECT": "him"}]},
	"Their regard of him as if he were an animal.": {
		"regard": [{"SBAR": "as if he were an animal", "SUBJECT": "Their", "OBJECT": "him"}]},

	# NOM-PP-THAT-S-SUBJUNCT (SUBJUNCT is ignored)
	"The loanshark suggested to Mimi that he will arrive on time.": {
		"suggested": [{"PP": "to Mimi", "SUBJECT": "The loanshark", "SBAR": "that he will arrive on time"}],
		"arrive": [{"SUBJECT": "he"}]},
	"The loanshark's suggestion to Mimi that he will arrive on time.": {
		"suggestion": [{"PP": "to Mimi", "SUBJECT": "The loanshark", "SBAR": "that he will arrive on time"}],
		"arrive": [{"SUBJECT": "he"}]},

	# NOM-PP-THAT-S
	"They admitted to the authorities that the children do drugs.": {
		"admitted": [{"PP": "to the authorities", "SUBJECT": "They", "SBAR": "that the children do drugs"}],
		"do": [{"SUBJECT": "the children", "OBJECT": "drugs"}],
		"authorities": [{"SUBJECT": "authorities"}]},
	"Their admission to the authorities that the children do drugs.": {
		"admission": [{"PP": "to the authorities", "SUBJECT": "Their", "SBAR": "that the children do drugs"}],
		"do": [{"SUBJECT": "the children", "OBJECT": "drugs"}],
		"authorities": [{"SUBJECT": "authorities"}]},

	# NOM-NP-S
	"I reminded her that the car had been stolen.": {
		"reminded": [{"SUBJECT": "I", "SBAR": "that the car had been stolen", "OBJECT": "her"}]},
	"My reminder to her that the car had been stolen.": {
		"reminder": [{"SUBJECT": "reminder", "SBAR": "that the car had been stolen", "OBJECT": "her"}]},

	# NOM-S-SUBJUNCT (SUBJUNCT is ignored)
	"I demanded that he be in tune.": {
		"demanded": [{"SUBJECT": "I", "SBAR": "that he be in tune"}]},
	"I specified that he should be on time.": {
		"specified": [{"SUBJECT": "I", "SBAR": "that he should be on time"}]},
	"The electrician recommended she buy extra insulation.": {
		"recommended": [{"SUBJECT": "The electrician", "SBAR": "she buy extra insulation"}],
		"buy": [{"SUBJECT": "she", "OBJECT": "extra insulation"}],
		"insulation": []},
	"The electrician's recommendation that she buy extra insulation.": {
		"recommendation": [{"SUBJECT": "The electrician", "SBAR": "that she buy extra insulation"}],
		"buy": [{"SUBJECT": "she", "OBJECT": "extra insulation"}],
		"insulation": []},

	# NOM-THAT-S
	"She observed that the world is better today.": {
		"observed": [{"SUBJECT": "She", "SBAR": "that the world is better today"}]},
	"Her observation that the world is better today.": {
		"observation": [{"SUBJECT": "Her", "SBAR": "that the world is better today"}]},

	# NOM-S
	"She knows John is a student.": {
		"knows": [{"SUBJECT": "She", "SBAR": "John is a student"}],
		"student": [{"SUBJECT": "student"}]},
	"Her knowledge that John is a student.": {
		"knowledge": [{"SUBJECT": "Her", "SBAR": "that John is a student"}],
		"student": [{"SUBJECT": "student"}]},

	########################################################################
	# AS-Phrases complements

	# NOM-NP-AS-ING
	"She diagnosed him as pretending to be sick.": {
		"diagnosed": [{"SUBJECT": "She", "OBJECT": "him", "ING": "pretending to be sick"}],
		"pretending": [{"TO-INF": "to be sick"}]},
	"Her diagnosis of him as pretending to be sick.": {
		"diagnosis": [{"SUBJECT": "Her", "OBJECT": "him", "ING": "pretending to be sick"}],
		"pretending": [{"TO-INF": "to be sick"}]},
	"She diagnosed him as being ill with the measles.": {
		"diagnosed": [{"SUBJECT": "She", "OBJECT": "him", "ING": "being ill with the measles"}]},
	"Her diagnosis of him as being ill with the measles.": {
		"diagnosis": [{"SUBJECT": "Her", "OBJECT": "him", "ING": "being ill with the measles"}]},

	# NOM-NP-AS-ADJP
	"The part-time workers do not consider their plight as statistical.": {
		"workers": [{"SUBJECT": "workers"}],
		"consider": [{"MODIFIER": "statistical", "SUBJECT": "The part-time workers", "OBJECT": "their plight"}]},
	"The part-time workers' consideration of their plight as statistical.": {
		"workers": [{"SUBJECT": "workers"}],
		"consideration": [{"MODIFIER": "statistical", "SUBJECT": "The part-time workers'", "OBJECT": "their plight"}]},
	"They characterized the play as beautiful.": {
		"characterized": [{"MODIFIER": "beautiful", "SUBJECT": "They", "OBJECT": "the play"}],
		"play": []},
	"The play's characterization by them as beautiful appeared all over the news.": {
		"play": [],
		"characterization": [{"MODIFIER": "beautiful", "OBJECT": "The play", "SUBJECT": "them"}],
		"appeared": [{"SUBJECT": "The play's characterization by them as beautiful"}]},
	"They viewed the problem as interesting.": {
		"viewed": [{"MODIFIER": "interesting", "SUBJECT": "They", "OBJECT": "the problem"}]},
	"Their view of the problem as interesting, was heard by many.": {
		"view": [{"MODIFIER": "interesting", "SUBJECT": "Their", "OBJECT": "the problem"}],
		"heard": [{"OBJECT": "Their view of the problem as interesting"}]},

	# NOM-NP-PP-AS-NP
	"They mentioned the call to me as a possible lead.": {
		"mentioned": [{"PP1": "to me", "PP": "as a possible lead", "SUBJECT": "They", "OBJECT": "the call"}],
		"call": [],
		"lead": [{"MODIFIER": "possible"}]},
	"Their mention of the call, to me as a possible lead.": {
		"mention": [{"PP1": "to me", "PP": "as a possible lead", "SUBJECT": "Their", "OBJECT": "the call"}],
		"call": [],
		"lead": [{"MODIFIER": "possible"}]},

	# NOM-NP-AS-NP-SC
	"He supervised the children as their babysitter.": {
		"supervised": [{"SUBJECT": "He", "PP": "as their babysitter", "OBJECT": "the children"}],
		"babysitter": [{"SUBJECT": "babysitter", "OBJECT": "their"}]},
	"His supervision of the children as their babysitter.": {
		"supervision": [{"SUBJECT": "His", "OBJECT": "the children", "PP": "as their babysitter"}],
		"babysitter": [{"SUBJECT": "babysitter", "OBJECT": "their"}]},

	# NOM-NP-AS-NP
	"They accepted him as a doctor.": {
		"accepted": [{"PP": "as a doctor", "SUBJECT": "They", "OBJECT": "him"}]},
	"Their acceptance of him as a doctor.": {
		"acceptance": [{"PP": "as a doctor", "SUBJECT": "Their", "OBJECT": "him"}]},

	# NOM-AS-NP
	"Lulu failed as a pastry cook.": {
		"failed": [{"SUBJECT": "Lulu", "PP": "as a pastry cook"}],
		"cook": [{"SUBJECT": "cook", "OBJECT": "pastry"}]},
	"Lulu's failure as a pastry cook.": {
		"failure": [{"SUBJECT": "Lulu", "PP": "as a pastry cook"}],
		"cook": [{"SUBJECT": "cook", "OBJECT": "pastry"}]},

	########################################################################
	# ADV and ADJ complements

	# NOM-ADVP-PP
	"He reacted positively to the news.": {
		"reacted": [{"PP": "to the news", "SUBJECT": "He", "MODIFIER": "positively"}]},
	"His positive reaction to the news.": {
		"reaction": [{"MODIFIER": "positive", "PP": "to the news", "SUBJECT": "His"}]},

	# NOM-NP-ADVP
	"They treated the boy there.": {
		"treated": [{"SUBJECT": "They", "OBJECT": "the boy", "MODIFIER": "there"}]},
	"Their treatment of the boy, there.": {
		"treatment": [{"SUBJECT": "Their", "OBJECT": "the boy", "MODIFIER": "there"}]},

	# NOM-ADVP
	"He looked slowly.": {
		"looked": [{"SUBJECT": "He", "MODIFIER": "slowly"}]},
	"His slow look.": {
		"look": [{"MODIFIER": "slow", "SUBJECT": "His"}]},

	# NP and PP
	"They moved their money from the apartment to this house.": {
		"moved": [{"PP1": "from the apartment", "PP2": "to this house", "SUBJECT": "They", "OBJECT": "their money"}]},
	"Their money's movement by them from the apartment to this house.": {
		"movement": [{"PP1": "from the apartment", "OBJECT": "Their money", "SUBJECT": "them", "PP2": "to this house"}]},

	# NOM-NP-TO-NP
	"The department allocates a computer to new students.": {
		"allocates": [{"IND-OBJ": "new students", "SUBJECT": "The department", "OBJECT": "a computer"}],
		"computer": [],
		"students": [{"SUBJECT": "students"}]},
	"The department's allocation of a computer to new students.": {
		"allocation": [{"IND-OBJ": "new students", "SUBJECT": "The department", "OBJECT": "a computer"}],
		"computer": [],
		"students": [{"SUBJECT": "students"}]},
	"The department allocated new students a computer.": {
		"allocated": [{"IND-OBJ": "new students", "SUBJECT": "The department", "OBJECT": "a computer"}],
		"students": [{"SUBJECT": "students"}],
		"computer": []},
	"The department allocates to new students a computer.": {
		"allocates": [{"IND-OBJ": "new students", "SUBJECT": "The department", "OBJECT": "a computer"}],
		"students": [{"SUBJECT": "students"}],
		"computer": []},

	# NOM-NP-FOR-NP
	"The chef prepared a breakfast, for the guest.": {
		"prepared": [
			{"IND-OBJ": "the guest", "SUBJECT": "The chef", "OBJECT": "a breakfast"},
			{"PP": "for the guest", "SUBJECT": "The chef", "OBJECT": "a breakfast"}]},
	"The chef's preparation of breakfast, for the guest.": {
		"preparation": [
			{"SUBJECT": "The chef", "IND-OBJ": "the guest", "OBJECT": "breakfast"},
			{"OBJECT": "The chef", "IND-OBJ": "the guest", "SUBJECT": "breakfast"}]},
	"The chef prepared them a breakfast.": {
		"prepared": [
			{"IND-OBJ": "them", "SUBJECT": "The chef", "OBJECT": "a breakfast"}]},
	"The breakfast's preparation for them, of the chef.": {
		"preparation": [
			{"SUBJECT": "The breakfast", "IND-OBJ": "them", "OBJECT": "the chef"},
			{"OBJECT": "The breakfast", "IND-OBJ": "them", "SUBJECT": "the chef"}]},
	"The chef prepared for the guest a breakfast.": {
		"prepared": [
			{"IND-OBJ": "the guest", "SUBJECT": "The chef", "OBJECT": "a breakfast"},
			{"PP": "for the guest", "SUBJECT": "The chef", "OBJECT": "a breakfast"}]},

	# NOM-NP-PP
	"They attributed the painting to Masaccio.": {
		"attributed": [
			{"PP": "to Masaccio", "SUBJECT": "They", "OBJECT": "the painting"}]},
	"Their attribution of the painting to Masaccio.": {
		"attribution": [
			{"PP": "to Masaccio", "SUBJECT": "Their", "OBJECT": "the painting"}]},

	# NOM-PP-PP
	"Nabisco competed against Nestles, for market share.": {
		"competed": [{"PP2": "for market share", "SUBJECT": "Nabisco", "PP1": "against Nestles"}],
		"share": [
			{"SUBJECT": "market"},
			{"OBJECT": "market"}]},
	"Nabisco's competition against Nestles, for market share.": {
		"competition": [{"PP2": "for market share", "SUBJECT": "Nabisco", "PP1": "against Nestles"}],
		"share": [{"SUBJECT": "market"}, {"OBJECT": "market"}]},

	# NOM-PP
	"He advised on the compromise.": {
		"advised": [{"PP": "on the compromise", "SUBJECT": "He"}],
		"compromise": []},
	"The compromise advice by him.": {
		"compromise": [],
		"advice": [{"PP": "compromise", "SUBJECT": "him"}]},

	# NOM-NP-NP
	"She envied him his car.": {
		"envied": [{"IND-OBJ": "him", "SUBJECT": "She", "OBJECT": "his car"}]
	},
	"Her envy of his car to him.": {
		"envy": [
			{"IND-OBJ": "him", "SUBJECT": "Her", "OBJECT": "his car"},
			{"IND-OBJ": "him", "OBJECT": "Her", "SUBJECT": "his car"}]},

	# NOM-NP
	"The boy prefers cupcakes.": {
		"prefers": [{"SUBJECT": "The boy", "OBJECT": "cupcakes"}]},
	"The boy's preference for cupcakes.": {
		"preference": [
			{"SUBJECT": "The boy", "OBJECT": "cupcakes"},
			{"PP": "for cupcakes", "SUBJECT": "The boy"},
			{"PP": "for cupcakes", "OBJECT": "The boy"}]},

	########################################################################
	# INTRANSITIVE

	# NOM-INTRANS-RECIP
	"The streets intersect.": {
		"intersect": [{"SUBJECT": "The streets"}]},
	"The intersection of the streets.": {
		"intersection": [{"SUBJECT": "the streets"}]},

	# NOM-INTRANS
	"He replied.": {
		"replied": [{"SUBJECT": "He"}]},
	"His reply.": {
		"reply": [{"SUBJECT": "His"}]}
}
