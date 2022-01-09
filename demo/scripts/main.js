define([
	'jquery',
	'tag',
	'axios',
	'intro.min'
], function(
	$,
	TAG,
	axios,
	introJs
) {
	function main() {
		const DEMO_URL = "http://127.0.0.1:5000/nomlexDemo/"; //"https://nlp.biu.ac.il/~avivwn/nomlexDemo/";
		const EXTRACT_ENDPOINT = DEMO_URL + "extract/";
		const MATCH_ENDPOINT = DEMO_URL + "match/";
		const RANDOM_ENDPOINT = DEMO_URL + "get_random_example/";
		const FEEDBACK_ENDPOINT = DEMO_URL + "feedback/";

		const CACHE_KEY = "cache-key";
		const SENTENCE_KEY = "sentence";
		const EXTRACTIONS_KEY = "extractions";
		const MATCHES_KEY = "matches";

		const PARSED_DATA_KEY = "parsed_data";
		const MENTIONS_BY_EVENT_KEY = "mentions_by_event";
		const SORTED_NOUN_EVENTS_KEY = "sorted_noun_events";
		const SORTED_VERB_EVENTS_KEY = "sorted_verb_events";
		const APPEARING_EVENTS_KEY = "appearing_events";

		const THEME_COLOR_KEY = "theme-color-mode";
		const DARK_THEME = "Dark";
		const LIGHT_THEME = "Light";

		const OPTIONS_KEY = "options";
		const DEFAULT_OPTIONS = {"extraction-based": "rule-based", "verbs-cb": true, "nouns-cb": true, "ud-cb": true, "postag-cb": true};

		const DEFAULT_EXAMPLE = "[AGENT Apple] [# appointed] [APPOINTEE Tim Cook] [TITLE as CEO]. The appointment of Tim Cook, by Apple as a CEO was expected.";
		const TAGGED_PREDICATE_PATTERN = "[#";

		// JQuery Selectors
		const SUBMIT_BTN = $("#submit-button");
		const RANDOM_BTN = $("#random-button");
		const THEME_BTN = $("#theme-button");
		const FEEDBACK_BTN = $("#feedback-button");
		const GITHUB_BTN = $("#github-button");
		const EXTRACTIONS_CONSTRAINER = $("#extractions-container");
		const MATCHES_CONTAINER = $("#matches-container");
		const SENTENCE_INPUT = $("#sentence-input");
		const MAIN_CONTENT = $("#main-content");

		// ------------------------------------------------------------------------------------------
		// Display a tree over the input sentence. The tree relations are given in the data parameter
		function display_tree(extractions_info, options, container, bottom_tag_category, bottom_link_category, separate_lines_sentences) {
			let parsed_data = extractions_info[PARSED_DATA_KEY];
			const mentions_by_event = extractions_info[MENTIONS_BY_EVENT_KEY];
			const sorted_noun_events = extractions_info[SORTED_NOUN_EVENTS_KEY];
			const sorted_verb_events = extractions_info[SORTED_VERB_EVENTS_KEY];
			const include_verbs = options["verbs-cb"];
			const include_nouns = options["nouns-cb"];

			let appearing_events = extractions_info[APPEARING_EVENTS_KEY];

			if (parsed_data === null)
				return;

			// Be aware that I made a few changes to TAG's API
			const tag = TAG.tag({
				// The `container` parameter can take either the ID of the main element or
				// the main element itself (as either a jQuery or native object)
				container: container,

				// The initial data to load.
				data: parsed_data,
				format: "odin",

				// Overrides of default options
				options: {
					showTopMainLabel: true,
					showTopLinksOnMove: true,
					showTopArgLabels: true,
					showBottomMainLabel: true,
					showBottomLinksOnMove: true,
					showBottomArgLabels: false,
					bottomLinkCategory: bottom_link_category,
					topTagCategory: "none",
					bottomTagCategory: bottom_tag_category,
					rowEdgePadding: 13,
					linkCurveWidth: 5,
					wordPadding: 9,
					compactRows: true,

					// New options that I added
					custom_theme: localStorage.getItem(THEME_COLOR_KEY),
					wordAfterSentencePadding: 50, // Padding between sentences (according to "." that is separated by spaces)
					separateLinesSentences: separate_lines_sentences
				}
			});

			tag.redraw();

			tag.words.forEach((word) => {
				if (word.idx in mentions_by_event) {
					const sentence_id = mentions_by_event[word.idx][0]["sentence"]; // There should be at least one mention

					let is_shown_event = false;
					let text_color = "";
					let hover_text_color = "";

					if (sorted_noun_events[sentence_id].includes(word.idx) && include_nouns) {
						is_shown_event = true;
						text_color = "#64b34c";
						hover_text_color = "#509c49";
					}

					if (sorted_verb_events[sentence_id].includes(word.idx) && include_verbs) {
						is_shown_event = true;
						text_color = "#299eec";
						hover_text_color = "#247fc2";
					}

					if (is_shown_event) {
						word.svgText.node.style.fontWeight = "550";
						word.svgText.node.style.fill = text_color;

						word.svgText.on("mouseover", function () {
							word.svgText.node.style.fill = hover_text_color;
						});

						word.svgText.on("mouseout", function () {
							word.svgText.node.style.fill = text_color;
						});

						// Whenever an event will be clicked, the arguments of the event will appear
						word.svgText.on("dblclick", function () {
							event.preventDefault();

							// Change the event that should appear of that sentence, only if the user clicked on an invisible event
							if (word.idx !== appearing_events[sentence_id]) {
								appearing_events[sentence_id] = word.idx;
								extractions_info[APPEARING_EVENTS_KEY] = appearing_events;

								choose_relevant_mentions(extractions_info, options);
								display_all_trees(extractions_info, container, options, separate_lines_sentences);
							}
						});
					}
				}
			});
		}

		function display_all_trees(extractions_info, container, options, separate_lines_sentences) {
			const show_UD = options["ud-cb"];
			const show_POS = options["postag-cb"];

			// Empty the container with the given id
			container.empty();

			// Get the relevant tags
			const bottom_tag_category = show_POS? "POS":"none";
			const bottom_link_category = show_UD? "universal-basic":"none";

			// Save displayed extractions
			localStorage.setItem(extractions_info[CACHE_KEY], JSON.stringify(extractions_info));

			// Display the sentence with the wanted relations
			display_tree(extractions_info, options, container, bottom_tag_category, bottom_link_category, separate_lines_sentences);

			// Scroll down to the results
			// $('html,body').animate({
			// 	scrollTop: $("#scroll-to-here").offset().top
			// }, 800);
		}

		// ------------------------------------------------------------------------------------------
		// Parses the sentence using the python server

		function show_loading_icon(container) {
			let loading_icon = "img/loading_icon.gif";
			if (localStorage.getItem(THEME_COLOR_KEY) === "Dark")
				loading_icon = "img/loading_icon_dark.gif";

			container[0].innerHTML = "<img src='" + loading_icon + "' style='margin:auto;display:flex;'/>";
		}

		function get_most_informative_event(extractions_info, sentence_id, include_verbs, include_noms) {
			const mentions_by_event = extractions_info[MENTIONS_BY_EVENT_KEY];
			const sorted_noun_events = extractions_info[SORTED_NOUN_EVENTS_KEY][sentence_id];
			const sorted_verb_events = extractions_info[SORTED_VERB_EVENTS_KEY][sentence_id];

			const most_informative_verb = Object.keys(sorted_verb_events).length?  sorted_verb_events[0]: -1;
			const num_of_verb_arguments = most_informative_verb !== -1? Object.keys(mentions_by_event[most_informative_verb]).length: -1;

			const most_informative_noun = Object.keys(sorted_noun_events).length?  sorted_noun_events[0]: -1;
			const num_of_noun_arguments = most_informative_noun !== -1? Object.keys(mentions_by_event[most_informative_noun]).length: -1;

			if (include_verbs && include_noms) {
				if (num_of_verb_arguments > num_of_noun_arguments)
					return most_informative_verb;
				else
					return most_informative_noun;
			}
			else {
				if (include_verbs)
					return most_informative_verb;

				if (include_noms)
					return most_informative_noun;
			}

			return -1;
		}

		function choose_appearing_events(extractions_info, options) {
			const n_sentences = extractions_info[SORTED_VERB_EVENTS_KEY].length;

			const include_verbs = options["verbs-cb"];
			const include_noms = options["nouns-cb"];

			let appearing_events = [];
			for (let i = 0; i < n_sentences; i++)
				appearing_events.push(get_most_informative_event(extractions_info, i, include_verbs, include_noms));

			extractions_info[APPEARING_EVENTS_KEY] = appearing_events
		}

		function choose_relevant_mentions(extractions_info, options) {
			const parsed_data = extractions_info[PARSED_DATA_KEY];
			const mentions_by_event = extractions_info[MENTIONS_BY_EVENT_KEY];
			const sorted_noun_events = extractions_info[SORTED_NOUN_EVENTS_KEY];
			const sorted_verb_events = extractions_info[SORTED_VERB_EVENTS_KEY];

			if (!(APPEARING_EVENTS_KEY in extractions_info))
				choose_appearing_events(extractions_info, options)

			let appearing_events = extractions_info[APPEARING_EVENTS_KEY];

			const include_verbs = options["verbs-cb"];
			const include_noms = options["nouns-cb"];
			let relevant_mentions = [];

			for (let i = 0; i < appearing_events.length; i++) {
				let appearing_event_id = appearing_events[i];

				const is_verb_related = sorted_verb_events[i].includes(appearing_event_id);
				const is_noun_related = sorted_noun_events[i].includes(appearing_event_id);
				if (appearing_event_id === -1 || (is_verb_related && !include_verbs) || (is_noun_related && !include_noms)) {
					appearing_event_id = get_most_informative_event(extractions_info, i, include_verbs, include_noms);
					appearing_events[i] = appearing_event_id;
				}

				const mentions_of_event = appearing_event_id !== -1? mentions_by_event[appearing_event_id]: [];
				relevant_mentions = relevant_mentions.concat(mentions_of_event);
			}

			parsed_data["mentions"] = relevant_mentions;
			extractions_info[APPEARING_EVENTS_KEY] = appearing_events;
		}

		function show_matching_extractions(matches_info) {
			localStorage.setItem(MATCHES_KEY, matches_info);

			if (matches_info === null)
				matches_info = "[THING No relevant arguments and predicates] were [# specified] by [AGENT the user].";
			else if(Object.keys(matches_info[APPEARING_EVENTS_KEY]).length === 0)
				matches_info = "Couldn't find any matches.";

			if (typeof matches_info === "string") {
				if (localStorage.getItem(THEME_COLOR_KEY) === DARK_THEME)
					MATCHES_CONTAINER[0].innerHTML = "<h5 style='padding-left:15px;color:lightgray;'>" + matches_info + "</h5>";
				else
					MATCHES_CONTAINER[0].innerHTML = "<h5 style='padding-left:15px;color:black;'>" + matches_info + "</h5>";
			}
			else {
				const options = JSON.parse(localStorage.getItem(OPTIONS_KEY));
				let matches_options = Object.assign({}, DEFAULT_OPTIONS);
				matches_options["ud-cb"] = options["ud-cb"];
				matches_options["postag-cb"] = options["postag-cb"];
				display_all_trees(matches_info, MATCHES_CONTAINER, matches_options, true);
			}
		}

		async function parse_sentence(must_parse_again= false) {
			SENTENCE_INPUT[0].value = SENTENCE_INPUT[0].value !== "" ? SENTENCE_INPUT[0].value : DEFAULT_EXAMPLE;

			// Don't parse the sentence if it is the same as the last one with the same properties
			let extractions_info = localStorage.getItem(EXTRACTIONS_KEY);
			let options = JSON.parse(localStorage.getItem(OPTIONS_KEY));
			let is_new_extractions = must_parse_again || localStorage.getItem(SENTENCE_KEY) !== SENTENCE_INPUT[0].value || extractions_info == null;
			let has_tagged_info = SENTENCE_INPUT[0].value.includes(TAGGED_PREDICATE_PATTERN);

			localStorage.setItem(SENTENCE_KEY, String(SENTENCE_INPUT[0].value));

			if (has_tagged_info)
				show_loading_icon(MATCHES_CONTAINER);
			else
				show_matching_extractions(null);

			if (is_new_extractions) {
				show_loading_icon(EXTRACTIONS_CONSTRAINER);

				const extract_response = await axios.post(
					EXTRACT_ENDPOINT,
					{
						"text": SENTENCE_INPUT[0].value,
						"extraction-based": options["extraction-based"]
					}
				);

				extractions_info = extract_response.data;
				extractions_info[CACHE_KEY] = EXTRACTIONS_KEY;
				choose_relevant_mentions(extractions_info, options);
			}
			else
				extractions_info = JSON.parse(extractions_info);

			display_all_trees(extractions_info, EXTRACTIONS_CONSTRAINER, options, false);

			if (has_tagged_info) {
				const matches_response = await axios.post(
					MATCH_ENDPOINT,
					{
						"text": SENTENCE_INPUT[0].value,
						"extraction-based": options["extraction-based"],
					}
				);

				const matches_info = matches_response.data;
				matches_info[CACHE_KEY] = MATCHES_KEY;
				choose_relevant_mentions(matches_info, DEFAULT_OPTIONS);
				show_matching_extractions(matches_info);
			}
		}


		// ------------------------------------------------------------------------------------------
		// Handle theme color
		function change_theme_color(new_theme_color) {
			localStorage.setItem(THEME_COLOR_KEY, new_theme_color);

			// Toggle from dark mode to light mode for each element in the document
			const relevant_for_theme = [$("body"), $("header"), $("footer"),
				THEME_BTN, FEEDBACK_BTN, GITHUB_BTN, SUBMIT_BTN, RANDOM_BTN,
				EXTRACTIONS_CONSTRAINER, MATCHES_CONTAINER,
				SENTENCE_INPUT, MAIN_CONTENT];

			relevant_for_theme.forEach((e) => {
				if (e.hasClass("dark")) {
					if (new_theme_color === LIGHT_THEME)
						e.removeClass('dark');
				}
				else {
					if (new_theme_color === DARK_THEME)
						e.addClass('dark');
				}
			});

			// Toggle the text on the theme button, and the biu icon image
			const biu_icon = document.getElementById("biu-img");
			const biu_nlp_icon = document.getElementById("biu-nlp-img");
			if (new_theme_color === LIGHT_THEME) {
				THEME_BTN[0].innerHTML = DARK_THEME;
				biu_icon.src = "img/biu.png";
				biu_nlp_icon.src = "img/biu_nlp.png";
			}
			else {
				THEME_BTN[0].innerHTML = LIGHT_THEME;
				biu_icon.src = "img/biu_dark.png";
				biu_nlp_icon.src = "img/biu_nlp_dark.png";
			}
		}

		THEME_BTN.click(async (e) => {
			e.preventDefault();

			let theme_color_mode = DARK_THEME;
			if (THEME_BTN[0].innerHTML === LIGHT_THEME)
				theme_color_mode = LIGHT_THEME;

			change_theme_color(theme_color_mode);

			// Display extractions
			const options = JSON.parse(localStorage.getItem(OPTIONS_KEY));
			let extractions_info = JSON.parse(localStorage.getItem(EXTRACTIONS_KEY));
			display_all_trees(extractions_info, EXTRACTIONS_CONSTRAINER, options, false);

			// Display matches
			let matching_extractions_info = JSON.parse(localStorage.getItem(MATCHES_KEY));
			show_matching_extractions(matching_extractions_info);
		});



		// ------------------------------------------------------------------------------------------
		// When submit button was clicked, the current sentence is parsed with the current selected options
		SUBMIT_BTN.click(async (e) => {
			e.preventDefault();
			await parse_sentence();
		});

		// When the dice button was clicked, a random sentence is parsed with the current selected options
		RANDOM_BTN.click(async (e) => {
			e.preventDefault();

			const random_response = await axios.get(RANDOM_ENDPOINT);
			SENTENCE_INPUT[0].value = random_response.data["text"];

			await parse_sentence();
		});

		// When any checkbox (option) is checked or unchecked, the current sentence is parsed with the current options
		const triggers_for_parsing = [$('#rule-based'), $('#model-based'), $('#hybrid-based'), $('#verbs-cb'), $('#nouns-cb'), $('#ud-cb'), $('#postag-cb')];
		triggers_for_parsing.forEach((trigger) => {
			trigger.click(async (e) => {
				let options = JSON.parse(localStorage.getItem(OPTIONS_KEY));
				let must_parse_again = false;

				if (e.target.id.endsWith('based')) {
					if (options['extraction-based'] !== e.target.id)
						must_parse_again = true;

					options['extraction-based'] = e.target.id;
				} else
					options[e.target.id] = e.target.checked;

				localStorage.setItem(OPTIONS_KEY, JSON.stringify(options));

				if (must_parse_again)
					await parse_sentence(true);
				else {
					let extractions_info = JSON.parse(localStorage.getItem(EXTRACTIONS_KEY));
					choose_relevant_mentions(extractions_info, options);
					display_all_trees(extractions_info, EXTRACTIONS_CONSTRAINER, options, false);

					const matches_info = JSON.parse(localStorage.getItem(MATCHES_KEY));
					show_matching_extractions(matches_info);
				}
			});
		});



		// ------------------------------------------------------------------------------------------
		// If the feedback button was clicked, a feedback prompt is shown
		FEEDBACK_BTN.click(async (e) => {
			e.preventDefault();
			const feed = prompt("Please enter your feedback here:", "");
			if (feed != null)
			{
				let text_to_send = "";
				if ((feed !== ""))
					text_to_send = "User wrote:\n" + feed + "\n\n";
				text_to_send += "Last sentence input:\n" + SENTENCE_INPUT[0].value;
				await axios.post(FEEDBACK_ENDPOINT, {"text-to-send": text_to_send});
			}
		});



		// // ------------------------------------------------------------------------------------------
		// // If the into button was clicked, a full into is shown, using IntroJS.
		// const $intro_button = $("#intro-button");
		// $intro_button.click(async (e) => {
		//     e.preventDefault();
		//     introJs().start()
		// });

		// ------------------------------------------------------------------------------------------
		// Prepare the document to show its content, according to the last data or setentence and to the last theme
		$(document).ready(async function() {
			// Get an optional input from the url, after the last slash
			const slash_idx = window.location.href.lastIndexOf('/');
			if ((slash_idx + 1) !== window.location.href.length) {
				const sliced = decodeURI(window.location.href.slice(slash_idx + 1));

				const last_sentence = localStorage.getItem(SENTENCE_KEY);
				if (last_sentence !== sliced) {
					// The sentence should be parsed from skretch
					localStorage.setItem(SENTENCE_KEY, sliced);
					localStorage.removeItem(EXTRACTIONS_KEY);
				}
			}

			// Update the written input sentence, based on the last saved sentence
			if (localStorage.getItem(SENTENCE_KEY) !== null)
				SENTENCE_INPUT[0].value = localStorage.getItem("sentence");

			// The deafult theme is based on the system preferences
			if (localStorage.getItem(THEME_COLOR_KEY) == null)
				localStorage.setItem(THEME_COLOR_KEY, "System");

			// The default chosen options will be- all checkboxes are checked
			let options = localStorage.getItem(OPTIONS_KEY);
			if (options == null || JSON.stringify(Object.keys(JSON.parse(options))) !== JSON.stringify(Object.keys(DEFAULT_OPTIONS)))
				localStorage.setItem(OPTIONS_KEY, JSON.stringify(DEFAULT_OPTIONS));

			// Update the chosen options based on the last saved options
			options = JSON.parse(localStorage.getItem(OPTIONS_KEY));
			$('#verbs-cb').prop('checked', options["verbs-cb"]);
			$('#nouns-cb').prop('checked', options["nouns-cb"]);
			$('#ud-cb').prop('checked', options["ud-cb"]);
			$('#postag-cb').prop('checked', options["postag-cb"]);
			$('#' + options['extraction-based']).prop('checked', true);

			$('#body').removeClass('hide-all');

			// Theme based on the system preference (only if the user didn't changed the color already)
			if (localStorage.getItem(THEME_COLOR_KEY) === "System") {
				if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches)
					change_theme_color(DARK_THEME);
				else
					change_theme_color(LIGHT_THEME);
			} else
				change_theme_color(localStorage.getItem(THEME_COLOR_KEY));

			await parse_sentence();
		});
	}

	return main;
});