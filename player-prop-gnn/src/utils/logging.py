"""
Structured JSON Logging for Production
Phase 4.3 - Observability Infrastructure

Why JSON Logging:
- Machine-parseable (ELK, Datadog, CloudWatch)
- Searchable by structured fields
- Consistent format across services
- Easy to aggregate and analyze

Design Decisions:
- UTC timestamps (no timezone confusion)
- Request ID in every log (trace requests end-to-end)
- Extra fields via 'extra' dict in logger.info()
- No sensitive data (no passwords, API keys, PII)
"""
import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
import traceback


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs in JSON format.
    
    Output format:
    {
        "timestamp": "2025-11-16T10:30:45.123456Z",
        "level": "INFO",
        "logger": "src.api.main",
        "message": "Prediction completed",
        "request_id": "abc-123-def",
        "duration_ms": 78,
        "player_id": 5,
        "cache_hit": false
    }
    
    Thread-safe: logging module handles locking
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON string.
        
        Args:
            record: Standard logging.LogRecord
        
        Returns:
            JSON string with structured fields
        """
        # Base log data (always present)
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add request ID if present (from middleware)
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        
        # Add extra fields (passed via extra={} in logger calls)
        # Examples: duration_ms, player_id, cache_hit, etc.
        for key, value in record.__dict__.items():
            if key not in [
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'message', 'pathname', 'process', 'processName',
                'relativeCreated', 'thread', 'threadName', 'exc_info',
                'exc_text', 'stack_info', 'request_id'
            ]:
                # Add custom fields (e.g., duration_ms, player_id)
                log_data[key] = value
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': ''.join(traceback.format_exception(*record.exc_info))
            }
        
        # Serialize to JSON (compact format)
        try:
            return json.dumps(log_data, default=str)
        except Exception as e:
            # Fallback: If JSON serialization fails, return plain text
            return json.dumps({
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'level': 'ERROR',
                'message': f'Failed to serialize log: {e}',
                'original_message': str(record.getMessage())
            })


def setup_logging(
    level: str = 'INFO',
    format_json: bool = True,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure application-wide logging.
    
    Args:
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        format_json: If True, use JSON formatter. If False, use plain text.
        log_file: Optional file path for file logging (in addition to stdout)
    
    Returns:
        Root logger instance
    
    Side effects:
        - Configures root logger
        - Removes existing handlers (idempotent)
        - Adds stdout handler (always)
        - Adds file handler (if log_file provided)
    
    Example:
        logger = setup_logging(level='INFO', format_json=True)
        logger.info('API started', extra={'port': 8000})
    """
    # Get root logger
    logger = logging.getLogger()
    
    # Remove existing handlers (for idempotency)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set log level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Choose formatter
    if format_json:
        formatter = JSONFormatter()
    else:
        # Plain text formatter (for local development)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Add stdout handler (always)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    
    # Add file handler (optional)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.error(f'Failed to create file handler for {log_file}: {e}')
    
    return logger


class RequestLogger:
    """
    Helper for logging with request context.
    
    Usage:
        req_logger = RequestLogger(logger, request_id='abc-123')
        req_logger.info('Processing request', player_id=5)
    
    Automatically includes request_id in all logs.
    """
    
    def __init__(self, logger: logging.Logger, request_id: str):
        """
        Args:
            logger: Base logger instance
            request_id: Request ID to include in all logs
        """
        self.logger = logger
        self.request_id = request_id
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal helper that adds request_id to extra fields."""
        extra = {'request_id': self.request_id}
        extra.update(kwargs)
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """Log at DEBUG level with request context."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log at INFO level with request context."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log at WARNING level with request context."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log at ERROR level with request context."""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log at CRITICAL level with request context."""
        self._log(logging.CRITICAL, message, **kwargs)


# ============================================================================
# LOGGING BEST PRACTICES
# ============================================================================

"""
DO:
- Use structured logging: logger.info('Prediction completed', extra={'duration_ms': 78})
- Include context: player_id, request_id, cache_hit
- Log errors with exceptions: logger.error('Failed', exc_info=True)
- Use appropriate levels: DEBUG < INFO < WARNING < ERROR < CRITICAL

DON'T:
- Log sensitive data: passwords, API keys, personal info
- Log every request in production (too much volume)
- Use print() statements (not captured by logging infrastructure)
- Log in hot loops (performance impact)

LEVELS:
- DEBUG: Detailed diagnostic info (disabled in production)
- INFO: General informational messages (startup, shutdown, major operations)
- WARNING: Unexpected but recoverable issues (cache miss, slow query)
- ERROR: Errors that need attention (prediction failed, database down)
- CRITICAL: System failure (model won't load, can't start API)
"""