#!/usr/bin/python3.6
import sys
import os

# Add the project directory to the Python path
sys.path.insert(0, '/home/yourusername/Chan-New/web')

# Import the Flask app
from app import app as application

if __name__ == "__main__":
    application.run()
