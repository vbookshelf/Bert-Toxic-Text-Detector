from flask import Flask


app = Flask(__name__)


# import the views.py file from the app folder
from app import views