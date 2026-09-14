import logging

"""
Example usage for a verbose output:
import logging
from sgpykit.util.log import logger

logging.basicConfig(
    # see https://docs.python.org/3/library/logging.html#logrecord-attributes
    format="%(levelname)s %(name)s:%(lineno)d: %(message)s",
)
# switching log level for sgpykit. Also see https://docs.python.org/3/library/logging.html#logging-levels
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)
"""
logger = logging.getLogger("sgpykit")
logger.addHandler(logging.NullHandler())  # avoid "No handlers could be found" warnings


def set_logger_basic_format():
    """
    Set up basic logger output format.
    
    This is the most common format used in tutorials and examples, showing:
    - Log level (INFO, DEBUG, etc.)
    - Logger name and line number
    - The log message
    
    Example:
        >>> from sgpykit import set_logger_basic_format, set_logger_debug_level
        >>> set_logger_basic_format()  # Use default basic format
        >>> set_logger_debug_level()
    """
    format = "%(levelname)s %(name)s:%(lineno)d: %(message)s"
    logging.basicConfig(format=format)


def set_logger_custom_format(format):
    """
    Set a custom logger output format string.
    
    Allows complete customization of the log message format using Python's
    logging format specifiers.
    
    Args:
        format: Custom format string using Python logging format specifiers.
               See: https://docs.python.org/3/library/logging.html#logrecord-attributes
    
    Example:
        >>> from sgpykit import set_logger_custom_format
        >>> set_logger_custom_format("%(levelname)s: %(message)s")
    """
    logging.basicConfig(format=format)


def set_logger_info_level(logger_name="sgpykit"):
    """
    Set the logger level to INFO for a specific logger.
    
    INFO level will show informational messages, warnings, and errors,
    but not detailed debug messages.
    
    Args:
        logger_name: Name of the logger to configure. Defaults to "sgpykit".
    
    Example:
        >>> from sgpykit import set_logger_basic_format, set_logger_info_level
        >>> set_logger_basic_format()
        >>> set_logger_info_level()
    """
    log = logging.getLogger(logger_name)
    log.setLevel(logging.INFO)


def set_logger_debug_level(logger_name="sgpykit"):
    """
    Set the logger level to DEBUG for a specific logger.
    
    DEBUG level shows all messages including detailed debug information,
    useful for troubleshooting and development.
    
    Args:
        logger_name: Name of the logger to configure. Defaults to "sgpykit".
    
    Example:
        >>> from sgpykit import set_logger_basic_format, set_logger_debug_level
        >>> set_logger_basic_format()
        >>> set_logger_debug_level()
    """
    log = logging.getLogger(logger_name)
    log.setLevel(logging.DEBUG)


def set_logger_custom_level(level, logger_name="sgpykit"):
    """
    Set a custom logger level for a specific logger.
    
    Allows setting any Python logging level (logging.DEBUG, logging.INFO,
    logging.WARNING, logging.ERROR, logging.CRITICAL).
    
    Args:
        level: Python logging level constant (e.g., logging.INFO, logging.WARNING)
        logger_name: Name of the logger to configure. Defaults to "sgpykit".
    
    Example:
        >>> from sgpykit import set_logger_basic_format, set_logger_custom_level
        >>> import logging
        >>> set_logger_basic_format()
        >>> set_logger_custom_level(logging.WARNING)
    """
    log = logging.getLogger(logger_name)
    log.setLevel(level)
