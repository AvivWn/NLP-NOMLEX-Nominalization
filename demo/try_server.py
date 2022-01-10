import requests
from http import HTTPStatus

from demo.config import DEMO_CONFIG


SERVER_URL = f"http://127.0.0.1:{DEMO_CONFIG.PORT}/{DEMO_CONFIG.WWW_ENDPOINT}"


def demo_post_request(url):
	return requests.post(url, json={
		"text": text,
		"extraction-based": "rule-based"
	})


if __name__ == "__main__":
	text = "[AGENT Apple] [# appointed] [APPOINTEE Tim Cook] [TITLE as CEO]." \
		"The appointment of Tim Cook, by Apple as a CEO was expected."

	extract_endpoint = f"{SERVER_URL}/extract/"
	match_endpoint = f"{SERVER_URL}/match/"

	extract_response = demo_post_request(extract_endpoint)
	assert extract_response.status_code == HTTPStatus.OK

	match_response = demo_post_request(match_endpoint)
	assert match_response.status_code == HTTPStatus.OK
