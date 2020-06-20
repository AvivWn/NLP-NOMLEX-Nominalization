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

        function displayTree(data, containerId, category) {
            const container = $('#' + containerId);
            const basicTag = TAG.tag({
                // The `container` parameter can take either the ID of the main element or
                // the main element itself (as either a jQuery or native object)
                container: container,

                // The initial data to load.
                // Different formats might expect different types for `data`:
                // sE.g., the "odin" format expects the annotations as an
                // (already-parsed) Object, while the "brat" format expects them as a raw
                // String.
                // See the full documentation for details.
                data: data,
                format: "odin",

                // Overrides for default options
                options: {
                    showTopMainLabel: true,
                    showTopLinksOnMove: true,
                    showTopArgLabels: false,
                    showBottomMainLabel: true,
                    showBottomLinksOnMove: true,
                    showBottomArgLabels: true,
                    //topLinkCategory: "none",
                    BottomLinkCategory: "Universal-basic",
                    topTagCategory: "none",
                    bottomTagCategory: "POS",
                    //rowVerticalPadding: 2,
                    //compactRows: true,

                }
            });

            return basicTag
        }

        const $submitButton = $("#submitButton");

        $submitButton.click(async (e) => {
            e.preventDefault();

            // const $sentenceInput = $("#sentenceInput");
            // $sentenceInput[0].value = $sentenceInput[0].value != "" ? $sentenceInput[0].value : "The quick brown fox jumped over the lazy dog."
            //
            // const response = await axios.post('https://nlp.biu.ac.il/~avivwn/eud/annotate/', {sentence: $sentenceInput[0].value});

            var eUd = document.getElementById("GFG1").checked
            var eUdPP = document.getElementById("GFG2").checked
            var eUdBart = document.getElementById("GFG3").checked
            var iterations = document.getElementById("GFG4").value
            var removeEudInfo = document.getElementById("GFG5").checked
            var includeBartInfo = document.getElementById("GFG6").checked
            var limitIterations = document.getElementById("GFG7").checked
            var removeNodeAddingConvs = document.getElementById("GFG8").checked

            const $sentenceInput = $("#sentenceInput");
            $sentenceInput[0].value = $sentenceInput[0].value != "" ? $sentenceInput[0].value : "The quick brown fox jumped over the lazy dog."

            const response = await axios.post('https://nlp.biu.ac.il/~avivwn/eud/annotate/', {sentence: $sentenceInput[0].value, eud: eUd, eud_pp: eUdPP, eud_bart: eUdBart, conv_iterations: limitIterations ? iterations : "inf", remove_eud_info: removeEudInfo, include_bart_info: includeBartInfo, remove_node_adding_convs: removeNodeAddingConvs});

            $('#containerBasic').empty()
            $('#containerPlus').empty()

            tag1 = displayTree(response.data.basic, "containerBasic", "universal-basic");
            tag2 = displayTree(response.data.plus, "containerPlus", "universal-enhanced", tag1.links);
            iters = document.getElementById("iters")
            iters.value = response.data.conv_done

            $('html,body').animate({
                scrollTop: $("#scrollToHere").offset().top
            }, 800);
        });

        var slash_idx = window.location.href.lastIndexOf('/')
        if ((slash_idx + 1) != window.location.href.length)
        {
            sliced = window.location.href.slice(slash_idx + 1)
            $("#sentenceInput")[0].value = decodeURIComponent(sliced)
            submitButton.click(this);
        }

        const $showmeButton = $("#showmeButton");

        $showmeButton.click(async (e) => {
            e.preventDefault();
            introJs().start()
        });

        const $feedbackButton = $("#feedbackButton");

        $feedbackButton.click(async (e) => {
            e.preventDefault();
            var feed = window.prompt("Please enter your feedback here:", "You can simply press OK, we will receive the attested senence.")
            if (feed != null)
            {
                var textToSend = ""
                if ((feed != "") && (feed != "You can simply press OK, we will receive the attested senence."))
                {
                    textToSend = "User wrote: " + feed + "\n"
                }
                textToSend += "Last sentence input:\n" + $("#sentenceInput")[0].value
                const response = await axios.post('https://nlp.biu.ac.il/~avivwn/eud/feedback/', {text_to_send: textToSend});
            }
        });
    }

    return main;
});

function showLimit(checkbox) {
    if (checkbox.checked) {
        document.getElementById("GFG4").style.visibility = "visible"
    } else {
        document.getElementById("GFG4").style.visibility = "hidden"
    }
}

function selectExample(event) {
    var examples = {
        "Complement control": "I love talking to friends.",
        "Noun-modifying clauses(reduced relcl)": "The house they made.",
        "Noun-modifying clauses(acl+participle)": "A vision softly creeping, left its seeds.",
        "Noun-modifying clauses(acl+infinitive)": "I designed, the house to build.",
        "Adverbial clauses": "You shouldn't text me while driving.",
        "Apposition": "E.T., the Extraterrestrial, phones home.",
        "Modifiers in conjunction": "I fought and I won behind enemy lines.",
        "Possessive modifiers in conjunction": "My father and mother met here.",
        "Elaboration/Specification Clauses": "I like fruits such as apples.",
        "Compounds": "I used canola oil.",
        "Genitive Constructions": "Army of zombies.",
        "Passivization Alternation": "The Sheriff was shot by Bob.",
        "Hyphen reconstruction": "I work at a Seattle-Based company.",
        "Adjectival modifiers": "I see dead people.",
        "Copular Sentences": "Tomorrow is another day.",
        "Evidential reconstructions(w/o matrix)": "He seems from Britain.",
        "Evidential reconstructions(with matrix)": "You seem to fear heights.",
        "Evidential reconstructions(reported)": "My momma always said that energy equals mc^2.",
        "Aspectual constructions": "He started talking funny.",
        "Indexicals": "He wonders around in these woods here.",
        "Extended multi-word prepositions": "The child ran ahead of his mother."
    };
    if (event.target.value == "Use loaded examples")
    {
        document.getElementById("sentenceInput").value = ""
        var examplesSelector = document.getElementById("examples");
        examplesSelector.options[0].selected = true;
    }
    else
    {
        document.getElementById("sentenceInput").value = examples[event.target.value]
        document.getElementById("submitButton").click();
    }
}
