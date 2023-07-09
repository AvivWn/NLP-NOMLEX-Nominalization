from yet_another_verb.arguments_extractor.extraction import ExtractedArgument, Extraction, Extractions, \
	ExtractedArguments


def reconstruct_arg(arg: ExtractedArgument) -> ExtractedArgument:
	return ExtractedArgument(
		start_idx=arg.start_idx,
		end_idx=arg.end_idx,
		head_idx=arg.head_idx,
		arg_type=arg.arg_type, arg_tag=arg.arg_tag,
		fulfilled_constraints=arg.fulfilled_constraints,
		encoding=arg.encoding
	)


def reconstruct_args(args: ExtractedArguments) -> ExtractedArguments:
	new_args = []
	for arg in args:
		new_arg = reconstruct_arg(arg)
		new_args.append(new_arg)

	return new_args


def reconstruct_extraction(extraction: Extraction, args=None, undetermined_args=None) -> Extraction:
	words, predicate_idx, predicate_lemma = extraction.words, extraction.predicate_idx, extraction.predicate_lemma
	args = extraction.args if args is None else args
	undetermined_args = extraction.undetermined_args if undetermined_args is None else undetermined_args

	new_extraction = Extraction(
		words=words, predicate_idx=predicate_idx, predicate_lemma=predicate_lemma,
		predicate_postag=extraction.predicate_postag,
		args=reconstruct_args(args),
		undetermined_args=reconstruct_args(undetermined_args))
	return new_extraction


def reconstruct_extractions(extractions: Extractions) -> Extractions:
	return [reconstruct_extraction(ext) for ext in extractions]
