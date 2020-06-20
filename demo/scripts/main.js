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
        // ------------------------------------------------------------------------------------------
        // Display a tree over the input sentence. The tree relations are given in the data parameter
        function display_tree(data, containerId, bottomTagCategory, bottomLinkCategory) {
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
                    wordPadding: 10,
                    compactRows: true,

                    // New options that I added
                    custom_theme: localStorage.getItem("THEME-COLOR-MODE"),
                    wordAfterSentencePadding: 50 //Padding between sentences
                }
            });
            tag.redraw();
        }

        function display_all_trees(data) {
            const show_UD = document.getElementById("ud-cb").checked;
            const show_POS = document.getElementById("postag-cb").checked;

            // Empty the container that includes the last parsing results
            $('#rule-based-container').empty();

            // Get the relevant tags
            const bottomTagCategory = show_POS? "POS":"none";
            const bottomLinkCategory = show_UD? "universal-basic":"none";

            // Display the sentence with the wanted relations
            display_tree(data, "rule-based-container", bottomTagCategory, bottomLinkCategory);

            // Scroll down to the results
            $('html,body').animate({
                scrollTop: $("#scroll-to-here").offset().top
            }, 800);
        }

        // ------------------------------------------------------------------------------------------
        // Parses the sentence using the python server
        const $sentence_input = $("#sentence-input");

        async function parse_sentence() {
            // Get the checkbox information
            const include_verbs = document.getElementById("verbs-cb").checked;
            const include_noms = document.getElementById("noms-cb").checked;

            $sentence_input[0].value = $sentence_input[0].value !== "" ? $sentence_input[0].value : "The appointment of Tim Cook by Apple as a CEO was expected. Apple appointed Tim Cook as CEO.";


            // Send the information to the server and get a response
            const response = await axios.post(
                'https://nlp.biu.ac.il/~avivwn/nomlexDemo/annotate/',
                {
                    "sentence": $sentence_input[0].value,
                    "include_verbs": include_verbs,
                    "include_noms": include_noms
                }
            );

            // Save locally the last entered sentence and its parsing
            localStorage.setItem("LAST-DATA", JSON.stringify(response.data));
            localStorage.setItem("LAST-SENTENCE", $sentence_input[0].value);
            display_all_trees(response.data);
        }


        // ------------------------------------------------------------------------------------------
        // Handle theme color
        function change_theme_color(theme_button, new_theme_color) {
            localStorage.setItem("THEME-COLOR-MODE", new_theme_color);

            // Toggle from dark mode to light mode for each element in the document
            const relevant_for_theme = [$("body"), $("header"), $("footer"),
                                        $("#theme-button"), $("#feedback-button"), $("#github-button"),
                                        $("#rule-based-container"), $("#submit-button"), $("#sentence-input"), $("#main-content")];

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

            if (localStorage.getItem("LAST-DATA") == null) {
                parse_sentence();
            }
            else {
                display_all_trees(JSON.parse(localStorage.getItem("LAST-DATA")));
            }
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
        // When submit button was clicked, the current sentence is parsed with the current options
        const $submit_button = $("#submit-button");
        $submit_button.click(async (e) => {
            e.preventDefault();
            await parse_sentence();
        });

        // When any checkbox (option) is checked or unchecked, the current sentence is parsed with the current options
        const triggers_for_parsing = [$('#verbs-cb'), $('#noms-cb'), $('#ud-cb'), $('#postag-cb')];
        triggers_for_parsing.forEach((trigger) => {
            trigger.click(async (e) => {
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
            if ((slash_idx + 1) !== window.location.href.length)
            {
                const sliced = decodeURI(window.location.href.slice(slash_idx + 1));
                localStorage.setItem("LAST-SENTENCE", sliced);
                localStorage.removeItem("LAST-DATA");
            }

            // Update the last sentence, based on the last saved sentence
            if (localStorage.getItem("LAST-SENTENCE") != null) {
                $sentence_input[0].value = localStorage.getItem("LAST-SENTENCE");
            }

            // The deafult theme is based on the system preferences
            if (localStorage.getItem("THEME-COLOR-MODE") === null) {
                localStorage.setItem("THEME-COLOR-MODE", "System");
            }

            // Show the final html page
            $('#header').hide();
            $('#header').removeClass('hide-all');
            $('#main-content').removeClass('hide-all');
            $('#footer').removeClass('hide-all');
            $('#stam').removeClass('hide-all');

            // Theme based on the system preference (only if the user didn't changed the color already)
            if (localStorage.getItem("THEME-COLOR-MODE") === "System") {
                if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
                    change_theme_color(theme_button, "Dark");
                } else {
                    change_theme_color(theme_button, "Light");
                }
            } else {
                change_theme_color(theme_button, localStorage.getItem("THEME-COLOR-MODE"))
            }

            $('#header').show();
        });

    }

    return main;
});