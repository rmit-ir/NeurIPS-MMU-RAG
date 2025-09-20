import os
import posthog
from typing import Optional, Dict, Any

from tools.logging_utils import get_logger

logger = get_logger('posthog')


class PostHogService:
    """Service for PostHog analytics integration"""

    def __init__(self):
        self.client = None
        self.enabled = False
        self.api_key = os.getenv('POSTHOG_API_KEY')
        self.api_host = os.getenv('POSTHOG_HOST')
        self._initialize()

    def _initialize(self):
        """Initialize PostHog client if configuration is available"""
        if not self.api_key or not self.api_host:
            logger.info("PostHog not configured, analytics disabled")
            return

        try:
            posthog.api_key = self.api_key
            posthog.host = self.api_host
            self.enabled = True
            logger.info("PostHog analytics enabled")
        except Exception as e:
            logger.error("Failed to initialize PostHog", error=str(e))

    def capture(
        self,
        event: str,
        distinct_id: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """Capture an event in PostHog"""
        if not self.enabled:
            return

        try:
            posthog.capture(
                event=event,
                distinct_id=distinct_id,
                properties=properties or {}
            )
            logger.debug("PostHog event captured",
                         event_name=event, distinct_id=distinct_id)
        except Exception as e:
            logger.error("Failed to capture PostHog event",
                         error=str(e), event_name=event)


# Global instance
posthog_service = PostHogService()
