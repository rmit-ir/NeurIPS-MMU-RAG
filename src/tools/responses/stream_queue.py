"""
Stream queue mimicking Redis asyncio API.

MIGRATION TO REDIS:

To migrate this to use actual Redis, replace the StreamQueue class with redis.asyncio:

1. Install redis: pip install redis[hiredis]
2. Replace StreamQueue with Redis client:
   ```python
   import redis.asyncio as redis
   
   # Initialize Redis client
   redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
   
   # Replace methods:
   # - publish() -> redis_client.publish()
   # - subscribe() -> redis_client.subscribe() 
   # - exists() -> redis_client.exists()
   # - delete() -> redis_client.delete()
   # - set_active() -> redis_client.setex() for TTL
   ```
3. Use Redis Streams for message persistence:
   ```python
   # Add to stream: redis_client.xadd(channel, fields)
   # Read from stream: redis_client.xread({channel: '0'})
   ```
4. Redis handles TTL, persistence, and clustering automatically.
"""

import asyncio
import time
from typing import Dict, AsyncGenerator, Any, Callable
from collections import defaultdict, deque

from tools.logging_utils import get_logger


class StreamQueue:
    """Redis-like async queue for streaming responses."""

    def __init__(self, default_ttl: int):
        self._streams: Dict[str, deque] = {}  # Store messages persistently
        self._subscribers: Dict[str, set] = defaultdict(set)
        self._active: Dict[str, bool] = {}
        self._expiry: Dict[str, float] = {}  # TTL tracking

        self._streams_lock = asyncio.Lock()      # For stream data operations
        self._subscribers_lock = asyncio.Lock()  # For subscriber management
        self._metadata_lock = asyncio.Lock()     # For active/expiry metadata

        self.default_ttl = default_ttl
        self.logger = get_logger('stream_queue')

    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel. Returns number of subscribers."""
        # Use streams lock for stream data operations
        async with self._streams_lock:
            # Initialize channel if it doesn't exist
            if channel not in self._streams:
                self._streams[channel] = deque()

            # Store message persistently
            self._streams[channel].append(message)

        # Use subscribers lock for subscriber operations
        async with self._subscribers_lock:
            count = 0
            dead_subs = set()

            for sub_queue in self._subscribers[channel]:
                try:
                    sub_queue.put_nowait(message)
                    count += 1
                except asyncio.QueueFull:
                    # Skip full queues
                    pass
                except Exception:
                    dead_subs.add(sub_queue)

            # Clean up dead subscribers
            for dead_sub in dead_subs:
                self._subscribers[channel].discard(dead_sub)

            return count

    async def subscribe(self, channel: str) -> AsyncGenerator[Any, None]:
        """Subscribe to channel and yield messages."""
        self.logger.info("New subscriber", channel=channel, stats=self.stats())
        # Check expiry with metadata lock
        async with self._metadata_lock:
            # Check if channel has expired
            if channel in self._expiry and time.time() > self._expiry[channel]:
                await self._cleanup_expired_channel(channel)
                return

        sub_queue = asyncio.Queue()

        # Send existing messages and add subscriber with appropriate locks
        async with self._streams_lock:
            # Send all existing messages to new subscriber
            if channel in self._streams:
                self.logger.info("Sending existing messages to request queue",
                                 channel=channel)
                for message in self._streams[channel]:
                    try:
                        sub_queue.put_nowait(message)
                    except asyncio.QueueFull:
                        pass  # Skip if queue is full

        async with self._metadata_lock:
            # Only add to subscribers if stream is still active
            if channel in self._active and self._active[channel]:
                self.logger.info("Adding new subscriber to channel",
                                 channel=channel, active=self._active[channel])
                async with self._subscribers_lock:
                    self._subscribers[channel].add(sub_queue)

        try:
            self.logger.info("Subscribing and yielding messages",
                             channel=channel)
            while True:
                message = await sub_queue.get()
                if message is None:  # End signal
                    break
                yield message
        finally:
            async with self._subscribers_lock:
                self._subscribers[channel].discard(sub_queue)

    async def _cleanup_expired_channel(self, channel: str):
        """Clean up expired channel (internal method, assumes lock is held)."""
        self._active.pop(channel, None)
        self._streams.pop(channel, None)
        self._expiry.pop(channel, None)
        if channel in self._subscribers:
            del self._subscribers[channel]

    async def exists(self, channel: str) -> bool:
        """Check if channel exists and is active or cached."""
        # Check streams first
        async with self._streams_lock:
            has_stream_data = channel in self._streams

        # Check metadata (active status and expiry)
        async with self._metadata_lock:
            # Check if expired
            if channel in self._expiry and time.time() > self._expiry[channel]:
                await self._cleanup_expired_channel(channel)
                return False

            is_active = channel in self._active and self._active[channel]

            # Channel exists if it's active or has cached data
            exists = is_active or has_stream_data
            self.logger.info("Channel exists check", channel=channel,
                             exists=exists, is_active=is_active,
                             has_stream_data=has_stream_data)
            return exists

    async def set_active(self, channel: str, active: bool = True):
        """Mark channel as active/inactive with optional TTL."""
        if active:
            # Set active status and initialize stream if needed
            async with self._metadata_lock:
                self._active[channel] = True

            async with self._streams_lock:
                if channel not in self._streams:
                    self._streams[channel] = deque()
        else:
            # Set inactive status and expiry
            async with self._metadata_lock:
                self._active[channel] = False
                # Set expiry time when stream completes
                self._expiry[channel] = time.time() + self.default_ttl

            # Send end signal to all subscribers
            await self.publish(channel, None)

    async def delete(self, channel: str):
        """Delete channel and clean up."""
        # Clean up metadata first
        async with self._metadata_lock:
            self._active.pop(channel, None)
            self._expiry.pop(channel, None)

        # Clean up stream data
        async with self._streams_lock:
            self._streams.pop(channel, None)

        # Clean up subscribers and send end signals
        async with self._subscribers_lock:
            if channel in self._subscribers:
                # Send end signal to remaining subscribers, if any
                for sub_queue in self._subscribers[channel]:
                    try:
                        sub_queue.put_nowait(None)
                    except Exception:
                        pass
                del self._subscribers[channel]

    def stats(self) -> Dict[str, Any]:
        """Return stats about the current state of the StreamQueue."""
        return {
            "streams": len(self._streams),
            "channels": sorted(self._subscribers.keys()),
            "active_channels": sorted([k for k, v in self._active.items() if v]),
            "subscribers": {k: len(v) for k, v in self._subscribers.items()},
            "active": {k: v for k, v in self._active.items() if v},
            "expired": {k: v for k, v in self._expiry.items() if time.time() > v},
        }


# Global instance with 1 day TTL
stream_queue = StreamQueue(default_ttl=3600 * 24)


async def get_or_start_stream(
    channel: str,
    stream_factory: Callable[[], AsyncGenerator[Any, None]]
) -> AsyncGenerator[Any, None]:
    """
    Get existing stream or start new one.
    Redis-like interface for stream management.
    """
    # Check if stream already exists
    stream_exists = await stream_queue.exists(channel)

    if stream_exists:
        # Subscribe to existing stream
        async for message in stream_queue.subscribe(channel):
            yield message
    else:
        # Set up the stream first
        await stream_queue.set_active(channel, True)

        # Start producer task that will run independently
        async def producer():
            try:
                async for item in stream_factory():
                    await stream_queue.publish(channel, item)
            except Exception as e:
                await stream_queue.publish(channel, f"ERROR: {e}")
            finally:
                # Mark stream as inactive and send completion signal
                await stream_queue.set_active(channel, False)
                asyncio.create_task(cleanup_channel(channel))

        # Start the producer task
        producer_task = asyncio.create_task(producer())

        # Subscribe to the stream we just started
        try:
            async for message in stream_queue.subscribe(channel):
                yield message
        except Exception:
            # If subscription fails, make sure to cancel the producer
            producer_task.cancel()
            raise


async def cleanup_channel(channel: str):
    """Clean up channel after appropriate delay based on TTL."""
    await asyncio.sleep(stream_queue.default_ttl)
    await stream_queue.delete(channel)
