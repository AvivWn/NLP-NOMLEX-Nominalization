require.config({
	//By default load any module IDs from js/lib
	baseUrl: 'scripts'
});

require([
	'tag',
	'jquery',
	'main'
], function(
	TAG,
	$,
	main
) {
	window.SVG.prepare();

	// Main function
	$(async () => {
		main();

	});
});
