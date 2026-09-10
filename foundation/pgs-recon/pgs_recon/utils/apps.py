import logging
import time


class ANSICode:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def setup_logging(level=logging.INFO):
    msg_fmt = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
    dt_fmt = '%Y-%m-%d %H:%M:%S %Z'
    logging.basicConfig(level=level, format=msg_fmt, datefmt=dt_fmt)
    logging.getLogger().handlers[0].formatter.converter = time.gmtime
