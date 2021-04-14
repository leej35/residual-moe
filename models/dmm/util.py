import logging


def get_logger(log_file):
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=log_file, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(print)
    logging.getLogger('').addHandler(console)

    def log(s):
        logging.info(s)

    return log
