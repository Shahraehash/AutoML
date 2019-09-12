"""
AutoML

Launches the API server and allows access
using an Angular SPA.
"""

from flask import Flask, redirect, url_for

from api import api

APP = Flask(__name__, static_url_path='/ui')

@APP.route('/')
def load_ui():
    redirect(url_for('ui'), code=302)

if __name__ == "__main__":
    APP.run()
