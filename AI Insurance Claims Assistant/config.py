import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-openai-api-key-here')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')

# Application Configuration
APP_TITLE = "AI Insurance Claims Assistant"
APP_VERSION = "1.0.0"
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'

# Processing Configuration
MAX_DESCRIPTION_LENGTH = 2000
MIN_DESCRIPTION_LENGTH = 20
MAX_CLAIM_AMOUNT = 1000000
DEFAULT_PROCESSING_TIMEOUT = 30  # seconds

# Fraud Detection Thresholds
LOW_RISK_THRESHOLD = 0.3
MEDIUM_RISK_THRESHOLD = 0.7
HIGH_RISK_THRESHOLD = 0.9

# UI Configuration
ITEMS_PER_PAGE = 10
CHART_HEIGHT = 400
SIDEBAR_WIDTH = 300

# Database Configuration (for future use)
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///claims.db')

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = 'claims_app.log'

# Security Configuration
SESSION_TIMEOUT = 3600  # 1 hour in seconds
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB