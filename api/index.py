import os
import sys

# Add root directory to sys.path so app, blockchain, face_utils can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app
