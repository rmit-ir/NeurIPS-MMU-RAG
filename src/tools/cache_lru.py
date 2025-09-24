"""
Simple LRU Cache implementation for caching responses with TTL support.
"""

from typing import Any, Optional, NamedTuple
from collections import OrderedDict
import asyncio
import time


class CacheEntry(NamedTuple):
    """Cache entry containing value and expiration time."""
    value: Any
    expires_at: Optional[float]  # None means no expiration


class LRUCache:
    """
    LRU (Least Recently Used) Cache implementation with TTL support.
    """

    def __init__(self, max_size: int = 100):
        """
        Initialize LRU cache with maximum size.

        Args:
            max_size: Maximum number of items to store in cache
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache. Returns None if not found or expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        async with self.lock:
            if key not in self.cache:
                return None

            entry = self.cache[key]
            current_time = time.time()

            # Check if item has expired
            if entry.expires_at is not None and current_time > entry.expires_at:
                # Remove expired item
                del self.cache[key]
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return entry.value

    async def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Put item in cache with optional TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds. None means no expiration.
        """
        async with self.lock:
            # Calculate expiration time
            expires_at = None
            if ttl is not None and ttl > 0:
                expires_at = time.time() + ttl

            entry = CacheEntry(value=value, expires_at=expires_at)

            if key in self.cache:
                # Update existing key
                del self.cache[key]
            elif len(self.cache) >= self.max_size:
                # Remove least recently used item
                self.cache.popitem(last=False)

            self.cache[key] = entry

    async def clear(self) -> None:
        """Clear all items from cache."""
        async with self.lock:
            self.cache.clear()

    async def size(self) -> int:
        """Get current cache size."""
        async with self.lock:
            return len(self.cache)

    async def cleanup_expired(self) -> int:
        """
        Remove all expired items from cache.

        Returns:
            Number of expired items removed
        """
        async with self.lock:
            current_time = time.time()
            expired_keys = []

            for key, entry in self.cache.items():
                if entry.expires_at is not None and current_time > entry.expires_at:
                    expired_keys.append(key)

            for key in expired_keys:
                del self.cache[key]

            return len(expired_keys)
