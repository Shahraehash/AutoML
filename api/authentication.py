"""
Methods to handle authentication.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import jwt
from ldap3 import Server, Connection, SUBTREE
from flask import jsonify, request, abort

def ldap_login():
    """
    Authenticate a user using LDAP.
    """

    payload = json.loads(request.data)

    connection = Connection(
      Server(os.getenv('LDAP_SERVER')),
      user=payload['username'],
      password=payload['password']
    )

    if not connection.bind():
        return abort(401)
    else:
        connection.search(
          search_base=os.getenv('LDAP_BASE_DN'),
          search_filter='(sAMAccountName=' + payload['username'].split('@')[0] + ')',
          search_scope = SUBTREE,
          attributes=['objectGUID', 'givenName', 'sn', 'mail', 'memberOf']
        )

        if os.getenv('LDAP_REQUIRED_GROUP') and not any(os.getenv('LDAP_REQUIRED_GROUP') in item for item in connection.entries[0]['memberOf']):
            return abort(401)

        token = jwt.encode(
          {
            'iss': 'milo-ml',
            'aud': 'milo-ml',
            'sub': payload['username'],
            'iat': datetime.utcnow(),
            'exp': datetime.now(tz=timezone.utc) + timedelta(days=1),
            'uid': str(connection.entries[0]['objectGUID']).strip('{}'),
            'name': str(connection.entries[0]['givenName']) + ' ' + str(connection.entries[0]['sn']),
            'email': str(connection.entries[0]['mail'])
          },
          os.getenv('LDAP_AUTH_SECRET'),
          algorithm='HS256'
        )

        connection.unbind()
        return jsonify({'token': token})

def ldap_verify(token):
    """
    Verifies a JWT token provided after an LDAP authentication.
    """

    return jwt.decode(token, os.getenv('LDAP_AUTH_SECRET'), issuer='milo-ml', audience='milo-ml', algorithms=['HS256'])
