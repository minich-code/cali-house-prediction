import logging 
import os 
from datetime import datetime 

# Define the logfile name using the current date and time 
log_file_name = f"{datetime.now().strftime('%m_%d_%H_%M_%S')}.log"

# Create the path to the log file inside the logs directory 
log_file_path = os.path.join(os.getcwd(), "logs", log_file_name)

# Create the log directory if it does not exist 
os.makedirs(os.path.dirname(log_file_path), exist_ok = True)


# Configure the logging module 
logging.basicConfig(
    filename = log_file_path,
    level = logging.INFO,
    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"  # Define the log message format,
)

# # Test the logger 
# if __name__ == "__main__":
#     # Log info message 
#     logging.info("Logging into the system")