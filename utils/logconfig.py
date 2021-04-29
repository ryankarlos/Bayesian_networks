import logging

MESS_FORMAT = "%(levelname)s:%(filename)s:%(asctime)s:  %(message)s"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def root_logger():
    """
    Default root logger setting
    Returns
    -------
    """
    logging.basicConfig(datefmt="%m/%d/%Y %I:%M:%S", level=logging.DEBUG)

    LOG = logging.getLogger()
    return LOG


def module_logger(
    level=logging.INFO, filename=None, format=MESS_FORMAT, time=TIME_FORMAT
):
    """
    Creates a module level logger
    Parameters
    ----------
    level
    format
    filename

    Returns
    -------

    """
    LOG = logging.getLogger()
    LOG.setLevel(level)
    formatter = logging.Formatter(format, time)
    if filename is not None:
        fh = logging.FileHandler(filename)
        fh.setFormatter(formatter)
        LOG.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    LOG.addHandler(ch)
    return LOG
