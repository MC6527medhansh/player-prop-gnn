"""
FastAPI Middleware for Monitoring
Phase 4.3 - Request Tracking Infrastructure

Middleware Philosophy:
- Runs on EVERY request (automatic instrumentation)
- Transparent to endpoints (no code changes required)
- Fail-safe (errors in middleware shouldn't break requests)

Design Decisions:
- Metrics middleware: Track request count, latency, status codes
- Request ID middleware: Generate unique ID for tracing
- Order matters: Request ID first, then metrics (so metrics can use ID)
"""
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
import uuid
from typing import Callable
import logging

from src.api.metrics import (
    REQUEST_COUNT,
    REQUEST_DURATION,
    TimedContext
)

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Generate unique request ID for each request.
    
    Request ID:
    - Added to request.state.request_id
    - Added to response header: X-Request-ID
    - Used for log correlation
    
    Format: UUID4 (e.g., "a3f8b9c2-1234-5678-90ab-cdef01234567")
    Collision probability: ~10^-18 (negligible)
    
    Usage:
        In endpoint:
            request_id = request.state.request_id
        
        In logs:
            logger.info('Message', extra={'request_id': request_id})
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Generate request ID and attach to request/response.
        
        Args:
            request: Incoming request
            call_next: Next middleware/endpoint in chain
        
        Returns:
            Response with X-Request-ID header
        """
        # Generate unique ID
        request_id = str(uuid.uuid4())
        
        # Attach to request state (accessible in endpoints)
        request.state.request_id = request_id
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Even if request fails, add request_id to error logs
            logger.error(
                f"Request failed: {e}",
                extra={'request_id': request_id},
                exc_info=True
            )
            raise
        
        # Add to response header (for client-side debugging)
        response.headers['X-Request-ID'] = request_id
        
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Track request metrics automatically.
    
    Metrics tracked:
    - Request count (by method, endpoint, status code)
    - Request duration (histogram for percentiles)
    
    Design:
    - Runs AFTER RequestIDMiddleware (so request_id is available)
    - Records metrics even if request fails
    - Uses TimedContext for automatic duration tracking
    
    Edge cases:
    - Client disconnects: Still records metrics
    - Middleware exception: Catches and logs, doesn't crash API
    - Slow requests: Recorded accurately (histogram buckets handle outliers)
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Track request metrics and timing.
        
        Args:
            request: Incoming request
            call_next: Next middleware/endpoint in chain
        
        Returns:
            Response (unchanged)
        
        Side effects:
            - Increments REQUEST_COUNT
            - Records REQUEST_DURATION
        """
        # Get request metadata
        method = request.method
        path = request.url.path
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Start timer
        start_time = time.time()
        
        # Process request (may fail)
        status_code = 500  # Default if something goes wrong
        try:
            response = await call_next(request)
            status_code = response.status_code
            
        except Exception as e:
            # Request failed - still record metrics
            logger.error(
                f"Request processing failed",
                extra={
                    'request_id': request_id,
                    'method': method,
                    'path': path,
                    'error': str(e)
                },
                exc_info=True
            )
            raise
        
        finally:
            # Always record metrics (even on failure)
            duration = time.time() - start_time
            
            # Increment request counter
            try:
                REQUEST_COUNT.labels(
                    method=method,
                    endpoint=path,
                    status_code=status_code
                ).inc()
            except Exception as e:
                logger.warning(f"Failed to record request count: {e}")
            
            # Record request duration
            try:
                REQUEST_DURATION.labels(
                    method=method,
                    endpoint=path
                ).observe(duration)
            except Exception as e:
                logger.warning(f"Failed to record request duration: {e}")
            
            # Log request completion
            logger.info(
                f"{method} {path} {status_code}",
                extra={
                    'request_id': request_id,
                    'method': method,
                    'path': path,
                    'status_code': status_code,
                    'duration_ms': int(duration * 1000)
                }
            )
        
        return response


class ErrorLoggingMiddleware(BaseHTTPMiddleware):
    """
    Catch and log all unhandled exceptions.
    
    Purpose:
    - Ensure all errors are logged (even if endpoint forgets)
    - Provide consistent error response format
    - Don't expose internal errors to clients
    
    Design:
    - Runs LAST (outermost middleware)
    - Catches ALL exceptions (including those from other middleware)
    - Returns 500 with safe error message (no stack traces to client)
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Catch and log unhandled exceptions.
        
        Args:
            request: Incoming request
            call_next: Next middleware/endpoint in chain
        
        Returns:
            Response (or 500 error response if exception)
        """
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        try:
            return await call_next(request)
        
        except Exception as e:
            # Log full exception with stack trace
            logger.critical(
                f"Unhandled exception: {e}",
                extra={'request_id': request_id},
                exc_info=True
            )
            
            # Return safe error to client (don't leak internals)
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=500,
                content={
                    'error': 'InternalServerError',
                    'message': 'An unexpected error occurred',
                    'request_id': request_id,
                    'detail': 'Check server logs for details'
                }
            )


# ============================================================================
# MIDDLEWARE CONFIGURATION
# ============================================================================

def setup_middleware(app: ASGIApp):
    """
    Add all middleware to FastAPI app.
    
    Order (outside to inside):
    1. ErrorLoggingMiddleware (catches everything)
    2. RequestIDMiddleware (generates ID first)
    3. MetricsMiddleware (uses ID from #2)
    
    Args:
        app: FastAPI application instance
    
    Usage:
        app = FastAPI()
        setup_middleware(app)
    """
    # Add in reverse order (they wrap each other)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(ErrorLoggingMiddleware)