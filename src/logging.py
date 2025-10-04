import logging
from enum import StrEnum

LOG_FORMAT_DEBUG = "%(levelname)s:%(message)s:%(pathname)s:%(funcName)s:%(lineno)d"

class LogLevels(StrEnum):
    info = "INFO"
    warn = "WARN"
    error = "ERROR"
    debug = "DEBUG"

def configure_logging(level: LogLevels = LogLevels.error):
    level = str(level).upper()
    levels = [level.value for level in LogLevels]
    
    if level not in levels:
        logging.basicConfig(level=LogLevels.error)
        return
    
    if level == LogLevels.debug:
        logging.basicConfig(level=level, format=LOG_FORMAT_DEBUG)
    
    logging.basicConfig(level=level)