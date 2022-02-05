"""
Handle license related tasks
"""

from datetime import datetime
import os

import requests
from flask import jsonify, abort, request
from licensing.models import LicenseKey
from licensing.methods import Helpers
from werkzeug.exceptions import HTTPException

def activate():
    """Activates a provided license key"""

    result = requests.post(
        'https://cloud.api.milo-ml.com/licensing/activate',
        json={
            'machine_code': Helpers.GetMachineCode(),
            'license_code': request.get_json()['license_code']
        }
    )

    if result.ok:
        result = result.json()

        with open('data/licensefile.skm', 'w') as file:
            file.write(result['license'])

        with open('data/license.pub', 'w') as file:
            file.write(result['public_key'])

        return jsonify({'success': True})
    else:
        abort(400)

def get_license():
    """Get the license of the current installation"""

    if os.path.exists('data/licensefile.skm') and os.path.exists('data/license.pub'):
        with open('data/license.pub') as file:
            public_key = file.read()

        with open('data/licensefile.skm', 'r') as file:
            license_key = LicenseKey.load_from_string(public_key, file.read())

        if license_key is None:
            return None
        else:
            # If the license is educational, bypass the expiration date and hardware ID check
            if license_key.f3:
                return license_key

            # Check if the license is valid for the current machine
            elif Helpers.IsOnRightMachine(license_key) and license_key.expires >= datetime.now():
                return license_key
            else:
                return None
    else:
        return None

def is_license_valid():
    """Ensures a valid license is cached"""

    return True if get_license() else False

class PaymentRequired(HTTPException):
    """HTTP Error for invalid license"""
    code = 402
    description = 'No valid license detected'
