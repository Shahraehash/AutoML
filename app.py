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
    return redirect(url_for('static', filename='index.html'))


@APP.route('/run')
def run():
    api.find_best_model()

if __name__ == "__main__":
    APP.run()
