import logging 
import os 
from logging.handlers import RotatingFileHandler 
from datetime import datetime 
import sys 

#Constants for log configuration 
LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"  #jitni baar program chale utni baar new log file bana with timestamp
MAX_LOG_SIZE = 5 * 1024 * 1024  # if file storage exceeds 5MB, create a new file
BACKUP_COUNT = 3   #sirf 3 purani files rakho, baaki delete karo. Disk space waste na ho.

#Construct log file path 
root_dir = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))   #join karo parent folder and current file structure ko and uska ek clean abs path do
log_dir_path = os.path.join(root_dir, LOG_DIR)   #logs naam ka folder create hojayega root dir mein
os.makedirs(log_dir_path, exist_ok = True)
log_file_path = os.path.join(log_dir_path, LOG_FILE)


def configure_logger():
    """
    Configures logging with a rotating file handler and a console handler
    """
    
    #Create a custom logger
    logger = logging.getLogger()   #Root logger--> Kya actually meh capture karna hai-->Everything above DEBUG level
    logger.setLevel(logging.DEBUG)

    #define formatter 
    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    #file handler with rotation 
    file_handler = RotatingFileHandler(log_file_path, maxBytes = MAX_LOG_SIZE, backupCount = BACKUP_COUNT, encoding = "utf-8")  #these are handler loggers-->Produces output for everything above INFO level
    file_handler.setFormatter(formatter) 
    file_handler.setLevel(logging.INFO)

    #Console handler 
    console_handler = logging.StreamHandler(sys.stdout)   #these are handler loggers-->Produces output for everything above INFO level
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    #Add handlers to logger 
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    
#calling the function
configure_logger()
