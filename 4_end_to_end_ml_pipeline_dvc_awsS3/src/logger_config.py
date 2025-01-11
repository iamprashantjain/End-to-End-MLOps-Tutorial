import logging
import os
from datetime import datetime

# Generate a log filename with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the log directory path (logs)
log_path = os.path.join(os.getcwd(), "logs")

# Create the log directory if it doesn't exist
os.makedirs(log_path, exist_ok=True)

# Full file path for the log file
LOG_FILEPATH = os.path.join(log_path, LOG_FILE)

# Configure the logger
logging.basicConfig(level=logging.INFO, 
                    filename=LOG_FILEPATH, 
                    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")

# You can also get a logger instance (optional, but useful for custom loggers)
logger = logging.getLogger(__name__)
