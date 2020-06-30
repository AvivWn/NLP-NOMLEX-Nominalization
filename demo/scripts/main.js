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
		const DATA_KEY = "parsed-data";
		const SENTENCE_KEY = "sentence";
		const APPEARING_EVENTS_KEY = "appearing-events";
		const INFORMATIVE_EVENTS_KEY = "informative-events";
		const MENTIONS_BY_EVENTS_KEY = "mentions-by-events";
		const EVENT_BY_WORD_INDEX_KEY = "event-by-word-index";
		const THEME_COLOR_KEY = "theme-color-mode";
		const OPTIONS_KEY = "options";

		// ------------------------------------------------------------------------------------------
		// Display a tree over the input sentence. The tree relations are given in the data parameter
		function display_tree(data, appearing_events, mentions_by_events, event_by_word_index, options, containerId, bottomTagCategory, bottomLinkCategory) {
			if (data === null) {
				return;
			}

			const container = $('#' + containerId);

			// Be aware that I made a few changes to TAG's API
			const tag = TAG.tag({
				// The `container` parameter can take either the ID of the main element or
				// the main element itself (as either a jQuery or native object)
				container: container,

				// The initial data to load.
				data: data,
				format: "odin",

				// Overrides of default options
				options: {
					showTopMainLabel: true,
					showTopLinksOnMove: true,
					showTopArgLabels: true,
					showBottomMainLabel: true,
					showBottomLinksOnMove: true,
					showBottomArgLabels: false,
					bottomLinkCategory: bottomLinkCategory,
					topTagCategory: "none",
					bottomTagCategory: bottomTagCategory,
					rowEdgePadding: 13,
					linkCurveWidth: 5,
					wordPadding: 12,
					compactRows: true,

					// New options that I added
					custom_theme: localStorage.getItem(THEME_COLOR_KEY),
					wordAfterSentencePadding: 50 //Padding between sentences
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
								localStorage.setItem(APPEARING_EVENTS_KEY, JSON.stringify(appearing_events));
								parse_sentence();
							}
						});
					}
				}
			});
		}

		function display_all_trees(data, appearing_events, event_by_word_index, mentions_by_events, options) {
			const show_UD = options["ud-cb"];
			const show_POS = options["postag-cb"];

			// Empty the container that includes the last parsing results
			$('#rule-based-container').empty();

			// Get the relevant tags
			const bottomTagCategory = show_POS? "POS":"none";
			const bottomLinkCategory = show_UD? "universal-basic":"none";

			// Display the sentence with the wanted relations
			display_tree(data, appearing_events, event_by_word_index, mentions_by_events, options, "rule-based-container", bottomTagCategory, bottomLinkCategory);

			// Scroll down to the results
			$('html,body').animate({
				scrollTop: $("#scroll-to-here").offset().top
			}, 800);
		}

		// ------------------------------------------------------------------------------------------
		// Parses the sentence using the python server
		const $sentence_input = $("#sentence-input");

		function update_relevant_mentions(parsed_data, informative_events, mentions_by_events, appearing_events, options) {
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
			localStorage.setItem(APPEARING_EVENTS_KEY, JSON.stringify(appearing_events));
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

		function choose_appearing_events(informative_events, mentions_by_events, options) {
			const include_verbs = options["verbs-cb"];
			const include_noms = options["noms-cb"];

			let appearing_events = {};

			Object.keys(informative_events).forEach(function(sentence_id) {
				sentence_id = parseInt(sentence_id);
				appearing_events[sentence_id] = get_most_informative_event(informative_events[sentence_id], mentions_by_events[sentence_id], include_verbs, include_noms);
			});

			return appearing_events;
		}

		async function parse_sentence(random_sentence=false) {
			$sentence_input[0].value = $sentence_input[0].value !== "" ? $sentence_input[0].value : "The appointment of Tim Cook by Apple as a CEO was expected. Apple appointed Tim Cook as CEO.";

			const options = JSON.parse(localStorage.getItem(OPTIONS_KEY));

			// Don't parse the sentence if it is the same as the last one
			let parsed_data = localStorage.getItem(DATA_KEY);
			let appearing_events = localStorage.getItem(APPEARING_EVENTS_KEY);			// The appearing event-id for each sentence-id
			let mentions_by_events = localStorage.getItem(MENTIONS_BY_EVENTS_KEY);		// The relevant emntions for each event-id per sentence-id
			let informative_events = localStorage.getItem(INFORMATIVE_EVENTS_KEY);		// The event-id with the most informative information (max number of arguments) per sentence-id
			let event_by_word_index = localStorage.getItem(EVENT_BY_WORD_INDEX_KEY);	// The event-id, sentence-id, is-verb by the word index in the input string (that contains all the sentences together)

			if (localStorage.getItem(SENTENCE_KEY) !== $sentence_input[0].value || random_sentence ||
				parsed_data == null || appearing_events == null || informative_events == null || mentions_by_events == null || event_by_word_index == null) {
				// Send the information to the server and get a response
				const response = await axios.post(
					'https://nlp.biu.ac.il/~avivwn/nomlexDemo/annotate/',
					{
						"sentence": $sentence_input[0].value,
						"random": random_sentence
					}
				);

				$sentence_input[0].value = response.data[SENTENCE_KEY];
				parsed_data = response.data[DATA_KEY];
				mentions_by_events = response.data[MENTIONS_BY_EVENTS_KEY];
				informative_events = response.data[INFORMATIVE_EVENTS_KEY];
				event_by_word_index = response.data[EVENT_BY_WORD_INDEX_KEY];
				appearing_events = choose_appearing_events(informative_events, mentions_by_events, options);

				// Save locally the last entered sentence, its parsing and other properties
				localStorage.setItem(DATA_KEY, JSON.stringify(parsed_data));
				localStorage.setItem(SENTENCE_KEY, $sentence_input[0].value);
				localStorage.setItem(APPEARING_EVENTS_KEY, JSON.stringify(appearing_events));
				localStorage.setItem(INFORMATIVE_EVENTS_KEY, JSON.stringify(informative_events));
				localStorage.setItem(MENTIONS_BY_EVENTS_KEY, JSON.stringify(mentions_by_events));
				localStorage.setItem(EVENT_BY_WORD_INDEX_KEY, JSON.stringify(event_by_word_index));
			}
			else {
				parsed_data = JSON.parse(parsed_data);
				appearing_events = JSON.parse(appearing_events);
				mentions_by_events = JSON.parse(mentions_by_events);
				informative_events = JSON.parse(informative_events);
				event_by_word_index = JSON.parse(event_by_word_index);
			}

			update_relevant_mentions(parsed_data, informative_events, mentions_by_events, appearing_events, options);
			display_all_trees(parsed_data, appearing_events, mentions_by_events, event_by_word_index, options);
		}


		// ------------------------------------------------------------------------------------------
		// Handle theme color
		function change_theme_color(theme_button, new_theme_color) {
			localStorage.setItem(THEME_COLOR_KEY, new_theme_color);

			// Toggle from dark mode to light mode for each element in the document
			const relevant_for_theme = [$("body"), $("header"), $("footer"),
				$("#theme-button"), $("#feedback-button"), $("#github-button"),
				$("#rule-based-container"), $("#submit-button"), $("#random-button"), $("#sentence-input"), $("#main-content")];

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

			parse_sentence();
		}


		const $theme_button = $("#theme-button");
		const theme_button = document.getElementById("theme-button");

		$theme_button.click(async (e) => {
			e.preventDefault();

			var theme_color_mode = "Dark";

			if (theme_button.innerHTML === "Light") {
				theme_color_mode = "Light";
			}

			change_theme_color(theme_button, theme_color_mode);
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
		const triggers_for_parsing = [$('#verbs-cb'), $('#noms-cb'), $('#ud-cb'), $('#postag-cb')];
		triggers_for_parsing.forEach((trigger) => {
			trigger.click(async (e) => {
				let options = JSON.parse(localStorage.getItem(OPTIONS_KEY));
				options[e.target.id] = e.target.checked;
				localStorage.setItem(OPTIONS_KEY, JSON.stringify(options));

				await parse_sentence();
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

				const last_sentence = localStorage.getItem(SENTENCE_KEY);
				if (last_sentence !== sliced) {
					// The sentence should be parsed from skretch
					localStorage.setItem(SENTENCE_KEY, sliced);
					localStorage.removeItem(DATA_KEY);
				}
			}

			// Update the written input sentence, based on the last saved sentence
			if (localStorage.getItem(SENTENCE_KEY) != null) {
				$sentence_input[0].value = localStorage.getItem(SENTENCE_KEY);
			}

			// The deafult theme is based on the system preferences
			if (localStorage.getItem(THEME_COLOR_KEY) == null) {
				localStorage.setItem(THEME_COLOR_KEY, "System");
			}

			// The default chosen options will be- all checkboxes are checked
			let options = localStorage.getItem(OPTIONS_KEY);
			const default_options = {"verbs-cb": true, "noms-cb": true, "ud-cb": true, "postag-cb": true};
			if (options == null || JSON.stringify(Object.keys(JSON.parse(options))) !== JSON.stringify(Object.keys(default_options))) {
				localStorage.setItem(OPTIONS_KEY, JSON.stringify(default_options));
			}

			// Update the chosen options based on the last saved options
			options = JSON.parse(localStorage.getItem(OPTIONS_KEY));
			$('#verbs-cb').prop('checked', options["verbs-cb"]);
			$('#noms-cb').prop('checked', options["noms-cb"]);
			$('#ud-cb').prop('checked', options["ud-cb"]);
			$('#postag-cb').prop('checked', options["postag-cb"]);

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
		});

	}

	return main;
});