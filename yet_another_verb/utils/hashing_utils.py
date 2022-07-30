from hashlib import sha256


def consistent_hash(value: str) -> int:
	return int(sha256(bytes(value.encode('utf-8'))).hexdigest(), 16)
