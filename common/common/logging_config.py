"""Centralized logging configuration for AI agent frameworks testing."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class LoggingConfig:
    """Centralized logging configuration for the project."""

    def __init__(self, 
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
        """Initialize logging configuration.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path
            log_format: Log message format
        """
        self.log_level = getattr(logging, log_level.upper())
        self.log_file = log_file
        self.log_format = log_format
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Set up the logging configuration."""
        logging.basicConfig(
            level=self.log_level,
            format=self.log_format,
            handlers=self._get_handlers(),
            force=True
        )

    def _get_handlers(self) -> list:
        """Get logging handlers.

        Returns:
            List of logging handlers
        """
        handlers = []

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_formatter = logging.Formatter(self.log_format)
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)

        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(self.log_level)
            file_formatter = logging.Formatter(self.log_format)
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)

        return handlers

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a logger instance.

        Args:
            name: Logger name

        Returns:
            Configured logger instance
        """
        return logging.getLogger(name)

    @staticmethod
    def setup_framework_logging(framework_name: str, 
                               log_level: str = "INFO",
                               log_dir: str = "logs") -> logging.Logger:
        """Set up framework-specific logging.

        Args:
            framework_name: Name of the framework
            log_level: Logging level
            log_dir: Directory for log files

        Returns:
            Configured logger instance
        """
        Path(log_dir).mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{log_dir}/{framework_name}_{timestamp}.log"
        
        config = LoggingConfig(log_level=log_level, log_file=log_file)
        return config.get_logger(framework_name)


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None) -> None:
    """Set up global logging configuration.

    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    LoggingConfig(log_level=log_level, log_file=log_file)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Configured logger instance
    """
    return LoggingConfig.get_logger(name) 