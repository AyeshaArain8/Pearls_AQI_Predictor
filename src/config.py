import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Read values from .env
API_KEY = os.getenv("API_KEY")
CITY = os.getenv("CITY")
LATITUDE = os.getenv("LATITUDE")
LONGITUDE = os.getenv("LONGITUDE")