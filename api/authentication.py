"""
Methods to handle authentication.
"""

import json
import os

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
          search_base=os.getenv('LDAP_ROOT_DN'),
          search_filter='(sAMAccountName=' + payload['username'].split('@')[0] + ')',
          search_scope = SUBTREE,
          attributes=['objectGUID']
        )
        guid = str(connection.entries[0]['objectGUID']).strip('{}')
        connection.unbind()
        return jsonify({'token': guid})
