from collections import defaultdict

class Ext:
	arguments_extractor: ...
	min_arguments: ...


	def match_argument(self):
		pass

	def match_subcat(self, subcat):
		# Check the compatability of the candidates with the required arguments first
		subcat._check_arguments_compatibility(args_per_candidate, self.requires, argument_candidates, predicate,
											is_required=True)
		if len(args_per_candidate.keys()) < len(self.requires):
			return []

		# Then, check for the optional arguments
		self._check_arguments_compatibility(args_per_candidate, self.optionals, argument_candidates, predicate)

		# From possible arguments for each candidate, to possible extractions
		extractions = self._get_extractions(args_per_candidate, predicate, suitable_verb, arguments_predictor)

		return extractions

	def match_entry(self, word_entries):
		extractions = []

		# Match the arguments based on each subcat for this word entry
		for subcat_type in self.subcats.keys():
			extractions += self.subcats[subcat_type].match_arguments(argument_candidates, predicate, suitable_verb,
																	 arguments_predictor)

		# Match arguments also based on the "next" entry in the lexicon
		# Meaning, the aruguments properties of the same word with another sense
		if self.next is not None:
			extractions += self.next.match_arguments(argument_candidates, predicate, suitable_verb, arguments_predictor)

		return extractions

	def extract_arguments(self, sentence, return_dependency_tree=False, min_arguments=0, using_default=False,
						  transer_args_predictor=None, context_args_predictor=None,
						  specify_none=False, trim_arguments=True, verb_noun_matcher=None, limited_verbs=None,
						  predicate_indexes=None, return_single=False):

		dependency_tree = get_dependency_tree(sentence)

		for token in dependency_tree:
			word_entry = self.search_word()

			if not word_entry:
				continue

			candidates = ...
			predicate = ...
			extractions = self.match_entry


			# Try to find an appropriate entry for the current word token
			word_entry, suitable_verb = self.find(token, using_default, verb_noun_matcher, limited_verbs)
			if not suitable_verb:
				continue

			# Get the candidates for the arguments of this word (based on relevant direct links in the ud dependency tree)
			argument_candidates = get_argument_candidates(token,
														  include_nom=not self.is_verb and not context_args_predictor)

			# The word itself can also be an argument
			predicate = self._wrap_predicate(token, word_entry, context_args_predictor)

			# if config.DEBUG:
			# print(f"Candidates for {token.orth_}:", [candidate_token._.subtree_text if candidate_token != token else candidate_token.orth_ for candidate_token in argument_candidates])

			# Get all the possible extractions of this word
			# get_extractions = timeit(word_entry.match_arguments)
			extractions = word_entry.match_arguments(argument_candidates, predicate, suitable_verb,
													 context_args_predictor)

			# Choose the most informative extractions
			extractions = self._choose_informative(extractions, predicate, suitable_verb, transer_args_predictor)

			for extraction in extractions:
				if len(extraction.get_complements()) >= min_arguments:
					extraction_dict = extraction.as_span_dict(trim_arguments)

					if extraction_dict == {}:
						continue

					self._update_unused_candidates(argument_candidates, token, extraction.get_tokens(), extraction_dict,
												   specify_none, trim_arguments)
					extractions_per_word[token].append(extraction_dict)

			if config.DEBUG and len(extractions_per_word.get(token, [])) > 1:
				pass
		# print(extractions_per_word[token])
		# raise Exception("Found word with more than one legal extraction.")

		return extractions_per_word
		# Extract arguments based on the verbal lexicon
		# extractions_per_verb = self.verb_lexicon.extract_arguments(dependency_tree, min_arguments, using_default, transer_args_predictor, context_args_predictor,
		#														   specify_none, trim_arguments, verb_noun_matcher, limited_verbs, predicate_indexes)
		extractions_per_verb = {}

		# Extract arguments based on the nominal lexicon
		extractions_per_nom = self.nom_lexicon.extract_arguments()

		if return_single:
			extractions_per_word = extractions_per_verb
			extractions_per_word.update(extractions_per_nom)

			if return_dependency_tree:
				return extractions_per_word, dependency_tree
			else:
				return extractions_per_word

		if return_dependency_tree:
			return extractions_per_verb, extractions_per_nom, dependency_tree
		else:
			return extractions_per_verb, extractions_per_nom