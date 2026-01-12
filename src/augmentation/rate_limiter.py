"""
Azure OpenAI Rate Limiter with Token Bucket Algorithm.

This module provides rate limiting for Azure OpenAI API calls,
implementing both RPM (requests per minute) and TPM (tokens per minute) limits.

Features:
- Token bucket algorithm for smooth rate limiting
- Exponential backoff with jitter for 429 errors
- Configurable limits from environment or parameters
- Metrics tracking for logging and monitoring
"""

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional, TypeVar, Tuple

from openai import RateLimitError

T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass
class RateLimiterMetrics:
    """Metrics tracked by the rate limiter."""
    total_requests: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retries: int = 0
    total_wait_time: float = 0.0
    start_time: float = field(default_factory=time.time)
    # Concurrent request tracking
    concurrent_requests: int = 0
    max_concurrent_requests: int = 0
    # Call duration tracking for ETA calculation
    call_durations: list = field(default_factory=list)
    max_duration_samples: int = 50  # Keep rolling average of last N calls
    
    @property
    def avg_call_duration(self) -> float:
        """Get average call duration for ETA calculation."""
        if not self.call_durations:
            return 0.0
        return sum(self.call_durations) / len(self.call_durations)
    
    def record_call_duration(self, duration: float) -> None:
        """Record a call duration for ETA tracking."""
        self.call_durations.append(duration)
        # Keep only the last N samples
        if len(self.call_durations) > self.max_duration_samples:
            self.call_durations = self.call_durations[-self.max_duration_samples:]
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary for logging."""
        elapsed = time.time() - self.start_time
        return {
            "total_requests": self.total_requests,
            "prompt_tokens": self.prompt_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "retries": self.retries,
            "total_wait_time_sec": round(self.total_wait_time, 2),
            "elapsed_sec": round(elapsed, 2),
            "avg_requests_per_min": round(self.total_requests / max(elapsed / 60, 0.001), 2),
            "avg_tokens_per_min": round(self.total_tokens / max(elapsed / 60, 0.001), 2),
            "concurrent_requests": self.concurrent_requests,
            "max_concurrent_requests": self.max_concurrent_requests,
            "avg_call_duration_sec": round(self.avg_call_duration, 3),
        }


class TokenBucket:
    """
    Token bucket for rate limiting.
    
    Tokens are added at a constant rate up to a maximum capacity.
    Each request consumes tokens; if not enough tokens are available,
    the request must wait.
    """
    
    def __init__(self, capacity: float, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum tokens the bucket can hold
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: float = 1.0) -> float:
        """
        Acquire tokens from the bucket, waiting if necessary.
        
        Args:
            tokens: Number of tokens to acquire
        
        Returns:
            Time waited in seconds
        """
        async with self._lock:
            wait_time = 0.0
            
            # Refill tokens based on elapsed time
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            
            # Wait if not enough tokens
            if self.tokens < tokens:
                deficit = tokens - self.tokens
                wait_time = deficit / self.refill_rate
                logger.debug(f"Rate limit: waiting {wait_time:.2f}s for {tokens} tokens")
                await asyncio.sleep(wait_time)
                self.tokens = min(self.capacity, self.tokens + wait_time * self.refill_rate)
                self.last_refill = time.monotonic()
            
            self.tokens -= tokens
            return wait_time


class AzureRateLimiter:
    """
    Rate limiter for Azure OpenAI API with dual bucket (RPM + TPM).
    
    Implements token bucket algorithm for smooth rate limiting and
    exponential backoff with jitter for handling 429 errors.
    """
    
    def __init__(
        self,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        max_retries: Optional[int] = None,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: RPM limit (default from env or 60)
            tokens_per_minute: TPM limit (default from env or 90000)
            max_retries: Max retry attempts (default from env or 5)
            base_delay: Initial retry delay in seconds
            max_delay: Maximum retry delay in seconds
        """
        # Load from environment with fallbacks
        self.rpm = requests_per_minute or int(os.getenv("AZURE_OPENAI_RPM", "60"))
        self.tpm = tokens_per_minute or int(os.getenv("AZURE_OPENAI_TPM", "90000"))
        self.max_retries = max_retries or int(os.getenv("AZURE_OPENAI_MAX_RETRIES", "5"))
        self.base_delay = base_delay
        self.max_delay = max_delay
        
        # Create token buckets
        # RPM bucket: capacity = RPM, refill = RPM/60 per second
        self._request_bucket = TokenBucket(
            capacity=self.rpm,
            refill_rate=self.rpm / 60.0
        )
        
        # TPM bucket: capacity = TPM, refill = TPM/60 per second
        self._token_bucket = TokenBucket(
            capacity=self.tpm,
            refill_rate=self.tpm / 60.0
        )
        
        # Metrics tracking
        self._metrics = RateLimiterMetrics()
        self._lock = asyncio.Lock()
        
        logger.info(
            f"Rate limiter initialized: RPM={self.rpm}, TPM={self.tpm}, "
            f"max_retries={self.max_retries}"
        )
    
    @property
    def metrics(self) -> dict:
        """Get current metrics as dictionary."""
        return self._metrics.to_dict()
    
    @property
    def concurrent_requests(self) -> int:
        """Get current number of concurrent requests."""
        return self._metrics.concurrent_requests
    
    @property
    def max_concurrent(self) -> int:
        """Get maximum concurrent requests seen."""
        return self._metrics.max_concurrent_requests
    
    @property
    def avg_call_duration(self) -> float:
        """Get average call duration in seconds."""
        return self._metrics.avg_call_duration
    
    async def acquire(self, estimated_tokens: int = 100) -> float:
        """
        Acquire rate limit allowance for a request.
        
        Waits until both RPM and TPM limits allow the request.
        
        Args:
            estimated_tokens: Estimated token count for the request
        
        Returns:
            Total wait time in seconds
        """
        # Acquire from both buckets
        request_wait = await self._request_bucket.acquire(1)
        token_wait = await self._token_bucket.acquire(estimated_tokens)
        
        total_wait = request_wait + token_wait
        
        async with self._lock:
            self._metrics.total_wait_time += total_wait
        
        return total_wait
    
    async def record_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0, cached_tokens: int = 0, success: bool = True) -> None:
        """
        Record actual token usage after a request completes.
        
        Args:
            prompt_tokens: Actual prompt tokens used (including cached)
            completion_tokens: Actual completion tokens used
            cached_tokens: Tokens served from cache
            success: Whether the request was successful
        """
        async with self._lock:
            self._metrics.total_requests += 1
            self._metrics.prompt_tokens += prompt_tokens
            self._metrics.cached_prompt_tokens += cached_tokens
            self._metrics.completion_tokens += completion_tokens
            self._metrics.total_tokens += (prompt_tokens + completion_tokens)
            if success:
                self._metrics.successful_requests += 1
            else:
                self._metrics.failed_requests += 1

    async def execute_with_retry(
        self,
        coro_factory: Callable[[], Coroutine[Any, Any, Tuple[T, int, int, int]]],
        estimated_tokens: int = 100,
        on_retry: Optional[Callable[[int, float, Exception], None]] = None,
    ) -> T:
        """
        Execute a coroutine with rate limiting and retry logic.
        
        Args:
            coro_factory: Factory function that creates the coroutine to execute.
                         Must return (result, prompt_tokens, completion_tokens, cached_tokens).
                         Called fresh on each retry attempt.
            estimated_tokens: Estimated tokens for rate limiting
            on_retry: Optional callback(attempt, delay, error) on retry
        
        Returns:
            Result from the coroutine
        
        Raises:
            Exception: If all retries are exhausted
        """
        last_error: Optional[Exception] = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Wait for rate limit
                await self.acquire(estimated_tokens)
                
                # Track concurrent requests
                async with self._lock:
                    self._metrics.concurrent_requests += 1
                    self._metrics.max_concurrent_requests = max(
                        self._metrics.max_concurrent_requests,
                        self._metrics.concurrent_requests
                    )
                
                # Execute the request with timing
                call_start = time.monotonic()
                try:
                    result, p_tokens, c_tokens, ca_tokens = await coro_factory()
                finally:
                    call_duration = time.monotonic() - call_start
                    async with self._lock:
                        self._metrics.concurrent_requests -= 1
                        self._metrics.record_call_duration(call_duration)
                
                # Record success and actual usage
                await self.record_usage(prompt_tokens=p_tokens, completion_tokens=c_tokens, cached_tokens=ca_tokens, success=True)
                
                return result
                
            except RateLimitError as e:
                last_error = e
                
                async with self._lock:
                    self._metrics.retries += 1
                
                if attempt >= self.max_retries:
                    logger.error(f"Rate limit exceeded after {self.max_retries} retries")
                    await self.record_usage(success=False)
                    raise
                
                # Exponential backoff with jitter
                delay = min(
                    self.base_delay * (2 ** attempt) + random.uniform(0, 1),
                    self.max_delay
                )
                
                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{self.max_retries + 1}), "
                    f"retrying in {delay:.2f}s: {e}"
                )
                
                if on_retry:
                    on_retry(attempt, delay, e)
                
                await asyncio.sleep(delay)
                
            except Exception as e:
                last_error = e
                async with self._lock:
                    self._metrics.concurrent_requests -= 1
                await self.record_usage(success=False)
                logger.error(f"Request failed with non-retryable error: {e}")
                raise
        
        # Should not reach here, but just in case
        raise last_error if last_error else RuntimeError("Unexpected retry loop exit")
    
    def reset_metrics(self) -> None:
        """Reset all metrics to initial values."""
        self._metrics = RateLimiterMetrics()
        logger.debug("Rate limiter metrics reset")
    
    def estimate_tokens(self, text: str, chars_per_token: float = 4.0) -> int:
        """
        Estimate token count for a text string.
        
        This is a rough estimate; actual token count depends on the model.
        
        Args:
            text: Input text
            chars_per_token: Average characters per token (default 4)
        
        Returns:
            Estimated token count
        """
        return max(1, int(len(text) / chars_per_token))
    
    def estimate_request_tokens(
        self,
        prompt: str,
        max_completion_tokens: int = 128
    ) -> int:
        """
        Estimate total tokens for a request (prompt + completion).
        
        Args:
            prompt: The prompt text
            max_completion_tokens: Maximum expected completion tokens
        
        Returns:
            Estimated total tokens
        """
        prompt_tokens = self.estimate_tokens(prompt)
        return prompt_tokens + max_completion_tokens


def create_rate_limiter_from_env() -> AzureRateLimiter:
    """
    Create a rate limiter with settings from environment variables.
    
    Environment variables:
        AZURE_OPENAI_RPM: Requests per minute (default: 60)
        AZURE_OPENAI_TPM: Tokens per minute (default: 90000)
        AZURE_OPENAI_MAX_RETRIES: Maximum retry attempts (default: 5)
    
    Returns:
        Configured AzureRateLimiter instance
    """
    return AzureRateLimiter()
