"""
Handle license related tasks
"""

import jwt
import datetime

from flask import jsonify, abort, request

async def activate_license():
    try:
        activate_license_key(request.json['license_code'])
        return jsonify({'success': True})
    except Exception:
        abort(400)

def activate_license_key(license_key: str):
    validate_key(license_key)
    with open("data/license.key", "w") as license_file:
        license_file.write(license_key)

def check_license_key():
    with open("data/license.key", "r") as license_file:
        license_key = license_file.read()
    validate_key(license_key)


def validate_key(license_key: str):
    with open("public_key.pem", "rb") as key_file:
        public_key = key_file.read()

    payload = jwt.decode(license_key, public_key, algorithms=["RS256"])
    expiration_date = datetime.datetime.fromisoformat(payload["expiration_date"])

    if expiration_date < datetime.datetime.now():
        raise Exception("Invalid or expired license key.")
