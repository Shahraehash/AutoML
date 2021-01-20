"""
Handle license related tasks
"""

import os

import requests
from flask import jsonify, abort
from licensing.models import LicenseKey
from licensing.methods import Helpers
from werkzeug.exceptions import HTTPException

def activate(license_code):
    """Activates a provided license key"""

    global LICENSE

    result = requests.post(
        'https://us-central1-milo-ml.cloudfunctions.net/activate',
        json={'machine_code': Helpers.GetMachineCode(), 'license_code': license_code}
    )

    if result.ok:
        result = result.json()

        with open('data/licensefile.skm', 'w') as file:
            file.write(result['license'])

        with open('data/license.pub', 'w') as file:
            file.write(result['public_key'])

        LICENSE = parse_license(result['public_key'], result['license'])
        return jsonify({'success': True})
    else:
        abort(400)

def get_license():
    """Get the license of the current installation"""

    if os.path.exists('data/licensefile.skm') and os.path.exists('data/license.pub'):
        with open('data/license.pub') as file:
            public_key = file.read()

        with open('data/licensefile.skm', 'r') as file:
            license_key = parse_license(public_key, file.read())

            if license_key is None or not Helpers.IsOnRightMachine(license_key):
                return None
            else:
                return license_key
    else:
        return None

def is_license_valid():
    """Ensures a valid license is cached"""

    return True if LICENSE else False

def parse_license(public_key, license_key):
    """Parses a license string into a license object"""

    return LicenseKey.load_from_string(public_key, license_key, 365)

class PaymentRequired(HTTPException):
    """HTTP Error for invalid license"""
    code = 402
    description = 'No valid license detected'

LICENSE = get_license()
