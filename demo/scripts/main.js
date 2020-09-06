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
		const SENTENCE_KEY = "sentence";
		const EXTRACTIONS_KEY = "extractions";
		const MATCHES_KEY = "matches";

		const DATA_KEY = "parsed-data";
		const INFORMATIVE_EVENTS_KEY = "informative-events";
		const MENTIONS_BY_EVENTS_KEY = "mentions-by-events";
		const EVENT_BY_WORD_INDEX_KEY = "event-by-word-index";
		const APPEARING_EVENTS_KEY = "appearing-events";

		const THEME_COLOR_KEY = "theme-color-mode";
		const OPTIONS_KEY = "options";

		const DEFAULT_OPTIONS = {"extraction-based": "rule-based", "verbs-cb": true, "noms-cb": true, "ud-cb": true, "postag-cb": true};

		// ------------------------------------------------------------------------------------------
		// Display a tree over the input sentence. The tree relations are given in the data parameter
		function display_tree(extractions_info, extractions_key, options, container_id, bottom_tag_category, bottom_link_category, separate_lines_sentences) {
			const parsed_data = extractions_info[DATA_KEY];
			const event_by_word_index = extractions_info[EVENT_BY_WORD_INDEX_KEY];
			let appearing_events = extractions_info[APPEARING_EVENTS_KEY];

			if (parsed_data === null) {
				return;
			}

			const container = $('#' + container_id);

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
					wordAfterSentencePadding: 50, //Padding between sentences (according to "." that is separated by spaces)
					separateLinesSentences: separate_lines_sentences
				}
			});

			tag.redraw();

			tag.words.forEach((word) => {
				if (word.idx in event_by_word_index) {
					const event_info = event_by_word_index[word.idx];
					const event_id = event_info["event-id"];
					const sentence_id = event_info["sentence-id"];
					const is_verb_related = event_info["is-verb"];
					const include_verbs = options["verbs-cb"];
					const include_noms = options["noms-cb"];

					if ((is_verb_related && include_verbs) || (!is_verb_related && include_noms)) {
						let text_color = "#299eec";
						let hover_text_color = "#247fc2";

						if (!is_verb_related) {
							text_color = "#64b34c";
							hover_text_color = "#509c49";
						}

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
							if (event_id !== appearing_events[sentence_id]) {
								appearing_events[sentence_id] = event_id;
								extractions_info[APPEARING_EVENTS_KEY] = appearing_events;

								if (extractions_key !== null) {
									sessionStorage.setItem(extractions_key, JSON.stringify(extractions_info));
								}

								update_relevant_mentions(extractions_info, options);
								display_all_trees(extractions_info, EXTRACTIONS_KEY, container_id, options, separate_lines_sentences);
							}
						});
					}
				}
			});
		}

		function display_all_trees(extractions_info, extractions_key, container_id, options, separate_lines_sentences) {
			const show_UD = options["ud-cb"];
			const show_POS = options["postag-cb"];

			// Empty the container with the given id
			$('#' + container_id).empty();

			// Get the relevant tags
			const bottom_tag_category = show_POS? "POS":"none";
			const bottom_link_category = show_UD? "universal-basic":"none";

			// Display the sentence with the wanted relations
			display_tree(extractions_info, extractions_key, options, container_id, bottom_tag_category, bottom_link_category, separate_lines_sentences);

			// Scroll down to the results
			// $('html,body').animate({
			// 	scrollTop: $("#scroll-to-here").offset().top
			// }, 800);
		}

		// ------------------------------------------------------------------------------------------
		// Parses the sentence using the python server
		const $sentence_input = $("#sentence-input");

		function update_relevant_mentions(extractions_info, options) {
			const parsed_data = extractions_info[DATA_KEY];
			const informative_events = extractions_info[INFORMATIVE_EVENTS_KEY];
			const mentions_by_events = extractions_info[MENTIONS_BY_EVENTS_KEY];
			let appearing_events = extractions_info[APPEARING_EVENTS_KEY];

			const include_verbs = options["verbs-cb"];
			const include_noms = options["noms-cb"];
			let all_relevant_mentions = [];

			Object.keys(appearing_events).forEach(function(sentence_id) {
				sentence_id = parseInt(sentence_id);
				let appearing_event_id = appearing_events[sentence_id];

				if (appearing_event_id !== -1) {
					let mentions_of_event = mentions_by_events[sentence_id][appearing_event_id];

					if (mentions_of_event !== []) {
						// The first mention is the event mention with the trigger
						const is_verb_related = mentions_of_event[0]["isVerbRelated"];

						if ((is_verb_related && !include_verbs) || (!is_verb_related && !include_noms)) {
							appearing_event_id = get_most_informative_event(informative_events[sentence_id], mentions_by_events[sentence_id], include_verbs, include_noms);

							if (appearing_event_id !== -1) {
								appearing_events[sentence_id] = appearing_event_id;
								mentions_of_event = mentions_by_events[sentence_id][appearing_event_id];
							}
							else {
								mentions_of_event = [];
							}
						}

						all_relevant_mentions = all_relevant_mentions.concat(mentions_of_event);
					}
				}
			});

			parsed_data["mentions"] = all_relevant_mentions;
			extractions_info[APPEARING_EVENTS_KEY] = appearing_events;
		}

		function get_most_informative_event(informative_events, mentions_by_events, include_verbs, include_noms) {
			const most_informative_verb = informative_events["verb"];
			const num_of_verb_arguments = most_informative_verb !== -1? Object.keys(mentions_by_events[most_informative_verb]).length: -1;
			const most_informative_nom = informative_events["nom"];
			const num_of_nom_arguments = most_informative_nom !== -1? Object.keys(mentions_by_events[most_informative_nom]).length: -1;

			if (include_verbs && include_noms) {
				if (num_of_verb_arguments > num_of_nom_arguments)
					return most_informative_verb;
				else
					return most_informative_nom;
			}
			else {
				if (include_verbs)
					return most_informative_verb;

				if (include_noms)
					return most_informative_nom;
			}

			return -1;
		}

		function choose_appearing_events(extractions_info, options) {
			const informative_events = extractions_info[INFORMATIVE_EVENTS_KEY];
			const mentions_by_events = extractions_info[MENTIONS_BY_EVENTS_KEY];

			const include_verbs = options["verbs-cb"];
			const include_noms = options["noms-cb"];

			let appearing_events = {};

			Object.keys(informative_events).forEach(function(sentence_id) {
				sentence_id = parseInt(sentence_id);
				appearing_events[sentence_id] = get_most_informative_event(informative_events[sentence_id], mentions_by_events[sentence_id], include_verbs, include_noms);
			});

			return appearing_events;
		}

		function show_matching_extractions(matches_info) {
			if (typeof matches_info !== "string") {
				matches_info[APPEARING_EVENTS_KEY] = choose_appearing_events(matches_info, DEFAULT_OPTIONS);
				update_relevant_mentions(matches_info, DEFAULT_OPTIONS);
				display_all_trees(matches_info, null, "matches-container", DEFAULT_OPTIONS, true);
				sessionStorage.setItem(MATCHES_KEY, JSON.stringify(matches_info));
			}
			else {
				if (localStorage.getItem(THEME_COLOR_KEY) === "Dark") {
					document.getElementById("matches-container").innerHTML = "<h5 style='padding-left:15px;color:lightgray;'>" + matches_info + "</h5>";
				} else {
					document.getElementById("matches-container").innerHTML = "<h5 style='padding-left:15px;color:black;'>" + matches_info + "</h5>";
				}
				sessionStorage.setItem(MATCHES_KEY, matches_info);
			}
		}

		async function parse_sentence(random_sentence=false, must_parse_again=false) {
			$sentence_input[0].value = $sentence_input[0].value !== "" ? $sentence_input[0].value : "[AGENT Apple] [# appointed] [APPOINTEE Tim Cook] [TITLE as CEO]. The appointment of Tim Cook by Apple as a CEO was expected.";

			// Don't parse the sentence if it is the same as the last one with the same properties
			let extractions_info = sessionStorage.getItem(EXTRACTIONS_KEY);
			let matches_info = sessionStorage.getItem(MATCHES_KEY);
			let options = JSON.parse(sessionStorage.getItem(OPTIONS_KEY));
			let is_new_extractions = must_parse_again || sessionStorage.getItem(SENTENCE_KEY) !== $sentence_input[0].value || random_sentence || extractions_info == null;

			// Put a loading icon in the containers, until the response receiving the response
			let loading_icon = "img/loading_icon.gif";
			if (localStorage.getItem(THEME_COLOR_KEY) === "Dark") {
				loading_icon = "img/loading_icon_dark.gif";
			}

			// Were predicates and arguments specified in the given sentence
			if ($sentence_input[0].value.includes("[#") && !random_sentence) {
				document.getElementById("matches-container").innerHTML = "<img src='" + loading_icon + "' style='margin:auto;display:flex;'/>";
			}
			else {
				matches_info = "[THING No relevant arguments and predicates] were [# specified] by [AGENT the user].";
				show_matching_extractions(matches_info)
			}

			if (is_new_extractions) {
				document.getElementById("extractions-container").innerHTML = "<img src='" + loading_icon + "' style='margin:auto;display:flex;'/>";

				// Send the information to the server and get the extractions
				const extractions_response = await axios.post(
					'https://nlp.biu.ac.il/~avivwn/nomlexDemo/extract/',
					{
						"sentence": $sentence_input[0].value,
						"random": random_sentence,
						"extraction-based": options["extraction-based"],
					}
				);

				$sentence_input[0].value = extractions_response.data[SENTENCE_KEY];
				extractions_info = extractions_response.data[EXTRACTIONS_KEY];
				extractions_info[APPEARING_EVENTS_KEY] = choose_appearing_events(extractions_info, options);

				// Save locally the last entered sentence and its parsing
				sessionStorage.setItem(EXTRACTIONS_KEY, JSON.stringify(extractions_info));
				sessionStorage.setItem(SENTENCE_KEY, $sentence_input[0].value);

				update_relevant_mentions(extractions_info, options);
				sessionStorage.setItem(EXTRACTIONS_KEY, JSON.stringify(extractions_info));
			}
			else {
				extractions_info = JSON.parse(extractions_info);
			}

			display_all_trees(extractions_info, EXTRACTIONS_KEY, "extractions-container", options, false);

			// New matches should be requested only if we can get any
			if ($sentence_input[0].value.includes("[#") && !random_sentence) {
				// Send the information to the server and get the founded matches
				const matches_response = await axios.post(
					'https://nlp.biu.ac.il/~avivwn/nomlexDemo/match/',
					{
						"extraction-based": options["extraction-based"],
						"extractions": extractions_info,
					}
				);
				matches_info = matches_response.data[MATCHES_KEY];
				show_matching_extractions(matches_info);
			}
		}


		// ------------------------------------------------------------------------------------------
		// Handle theme color
		function change_theme_color(theme_button, new_theme_color) {
			localStorage.setItem(THEME_COLOR_KEY, new_theme_color);

			// Toggle from dark mode to light mode for each element in the document
			const relevant_for_theme = [$("body"), $("header"), $("footer"),
				$("#theme-button"), $("#feedback-button"), $("#github-button"),
				$("#extractions-container"), $("#matches-container"), $("#submit-button"), $("#random-button"), $("#sentence-input"), $("#main-content")];

			relevant_for_theme.forEach((e) => {
				if (e.hasClass("dark")) {
					if (new_theme_color === "Light") {
						e.removeClass('dark');
					}
				}
				else {
					if (new_theme_color === "Dark") {
						e.addClass('dark');
					}
				}
			});

			// Toggle the text on the theme button, and the biu icon image
			const biu_icon = document.getElementById("biu-img");
			const biu_nlp_icon = document.getElementById("biu-nlp-img");
			if (new_theme_color === "Light") {
				theme_button.innerHTML = "Dark";
				biu_icon.src = "img/biu.png";
				biu_nlp_icon.src = "img/biu_nlp.png";
			}
			else {
				theme_button.innerHTML = "Light";
				biu_icon.src = "img/biu_dark.png";
				biu_nlp_icon.src = "img/biu_nlp_dark.png";
			}
		}

		function is_json_string(str) {
			try {
				JSON.parse(str);
			} catch (e) {
				return false;
			}
			return true;
		}

		const $theme_button = $("#theme-button");
		const theme_button = document.getElementById("theme-button");

		$theme_button.click(async (e) => {
			e.preventDefault();

			let theme_color_mode = "Dark";
			if (theme_button.innerHTML === "Light") {
				theme_color_mode = "Light";
			}

			change_theme_color(theme_button, theme_color_mode);

			// Display the current trees again
			const options = JSON.parse(sessionStorage.getItem(OPTIONS_KEY));
			let extractions_info = JSON.parse(sessionStorage.getItem(EXTRACTIONS_KEY));
			display_all_trees(extractions_info, EXTRACTIONS_KEY, "extractions-container", options, false);

			let matching_extractions_info = sessionStorage.getItem(MATCHES_KEY);
			if (is_json_string(matching_extractions_info)) {
				display_all_trees(JSON.parse(matching_extractions_info), null, "matches-container", DEFAULT_OPTIONS, true);
			}
			else {
				if (localStorage.getItem(THEME_COLOR_KEY) === "Dark") {
					document.getElementById("matches-container").innerHTML = "<h5 style='padding-left:15px;color:lightgray;'>" + matching_extractions_info + "</h5>";
				} else {
					document.getElementById("matches-container").innerHTML = "<h5 style='padding-left:15px;color:black;'>" + matching_extractions_info + "</h5>";
				}
			}
		});



		// ------------------------------------------------------------------------------------------
		// When submit button was clicked, the current sentence is parsed with the current selected options
		const $submit_button = $("#submit-button");
		$submit_button.click(async (e) => {
			e.preventDefault();
			await parse_sentence();
		});

		// When the dice button was clicked, a random sentence is parsed with the current selected options
		const $random_button = $("#random-button");
		$random_button.click(async (e) => {
			e.preventDefault();
			await parse_sentence(true);
		});

		// When any checkbox (option) is checked or unchecked, the current sentence is parsed with the current options
		const triggers_for_parsing = [$('#rule-based'), $('#model-based'), $('#hybrid-based'), $('#verbs-cb'), $('#noms-cb'), $('#ud-cb'), $('#postag-cb')];
		triggers_for_parsing.forEach((trigger) => {
			trigger.click(async (e) => {
				let options = JSON.parse(sessionStorage.getItem(OPTIONS_KEY));
				let must_parse_again = false;

				if (e.target.id.endsWith('based')) {
					options['extraction-based'] = e.target.id;
					must_parse_again = true
				} else {
					options[e.target.id] = e.target.checked;
				}

				sessionStorage.setItem(OPTIONS_KEY, JSON.stringify(options));

				if (must_parse_again) {
					await parse_sentence(false, true);
				}
				else {
					let extractions_info = JSON.parse(sessionStorage.getItem(EXTRACTIONS_KEY));
					update_relevant_mentions(extractions_info, options);
					sessionStorage.setItem(EXTRACTIONS_KEY, JSON.stringify(extractions_info));
					display_all_trees(extractions_info, EXTRACTIONS_KEY, "extractions-container", options, false);
				}
			});
		});



		// ------------------------------------------------------------------------------------------
		// If the feedback button was clicked, a feedback prompt is shown
		const $feedback_button = $("#feedback-button");
		$feedback_button.click(async (e) => {
			e.preventDefault();
			const feed = prompt("Please enter your feedback here:", "");
			if (feed != null)
			{
				var text_to_send = "";
				if ((feed !== ""))
				{
					text_to_send = "User wrote:\n" + feed + "\n\n";
				}
				text_to_send += "Last sentence input:\n" + $sentence_input[0].value;
				await axios.post('https://nlp.biu.ac.il/~avivwn/nomlexDemo/feedback/', {"text-to-send": text_to_send});
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
		$(document).ready(function() {
			// Get an optional input from the url, after the last slash
			const slash_idx = window.location.href.lastIndexOf('/');
			if ((slash_idx + 1) !== window.location.href.length) {
				const sliced = decodeURI(window.location.href.slice(slash_idx + 1));

				const last_sentence = sessionStorage.getItem(SENTENCE_KEY);
				if (last_sentence !== sliced) {
					// The sentence should be parsed from skretch
					sessionStorage.setItem(SENTENCE_KEY, sliced);
					sessionStorage.removeItem(EXTRACTIONS_KEY);
				}
			}

			// Update the written input sentence, based on the last saved sentence
			if (sessionStorage.getItem(SENTENCE_KEY) != null) {
				$sentence_input[0].value = sessionStorage.getItem(SENTENCE_KEY);
			}

			// The deafult theme is based on the system preferences
			if (localStorage.getItem(THEME_COLOR_KEY) == null) {
				localStorage.setItem(THEME_COLOR_KEY, "System");
			}

			// The default chosen options will be- all checkboxes are checked
			let options = sessionStorage.getItem(OPTIONS_KEY);
			if (options == null || JSON.stringify(Object.keys(JSON.parse(options))) !== JSON.stringify(Object.keys(DEFAULT_OPTIONS))) {
				sessionStorage.setItem(OPTIONS_KEY, JSON.stringify(DEFAULT_OPTIONS));
			}

			// Update the chosen options based on the last saved options
			options = JSON.parse(sessionStorage.getItem(OPTIONS_KEY));
			$('#verbs-cb').prop('checked', options["verbs-cb"]);
			$('#noms-cb').prop('checked', options["noms-cb"]);
			$('#ud-cb').prop('checked', options["ud-cb"]);
			$('#postag-cb').prop('checked', options["postag-cb"]);
			$('#' + options['extraction-based']).prop('checked', true);

			$('#body').removeClass('hide-all');

			// Theme based on the system preference (only if the user didn't changed the color already)
			if (localStorage.getItem(THEME_COLOR_KEY) === "System") {
				if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
					change_theme_color(theme_button, "Dark");
				} else {
					change_theme_color(theme_button, "Light");
				}
			} else {
				change_theme_color(theme_button, localStorage.getItem(THEME_COLOR_KEY))
			}

			parse_sentence();
		});

	}

	return main;
});