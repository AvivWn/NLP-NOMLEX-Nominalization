import requests


def demo_post_request(url):
	return requests.post(url, json={
		"text": text,
		"extraction-based": "rule-based"
	})


if __name__ == "__main__":
	text = "[AGENT Apple] [# appointed] [APPOINTEE Tim Cook] [TITLE as CEO]." \
		"The appointment of Tim Cook, by Apple as a CEO was expected."

	extract_endpoint = "http://127.0.0.1:5000/nomlexDemo/extract/"
	match_endpoint = "http://127.0.0.1:5000/nomlexDemo/match/"

	extract_response = demo_post_request(extract_endpoint)
	match_response = demo_post_request(match_endpoint)

	print(extract_response)
	print(match_response)
