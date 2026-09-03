"""Access to the chatterloop platform, over its developer API.

WHY NOT DIRECTLY, ANY MORE
--------------------------
This package used to hold a `MongoReader` and a `PostgresReader` and to open
connections to the platform's own datastores. The justification at the time was
that reads duplicate no logic and that no inbound service auth existed to go
through instead - a bot fails `jwtchecker` structurally, having no
`user_account` row and no device session.

Both halves of that have changed. `entity_token` is that inbound auth, and
developer_service is the door it opens. So the direct readers are gone - and so
is the direct Redis subscription the event consumer used to hold - and with
them:

  * three credentials this service no longer holds - two databases and the
    platform's Redis, the last of which could read every entity's channel and
    write to any of them;
  * the dependency on the platform's physical schema. Reading
    `newsfeed_comment.text` by hand meant a column rename in someone else's
    repo broke this service silently, at runtime, in a code path that only
    fires on a comment mention;
  * the pipeline's ability to read anything it could name. It now reads what a
    scoped token permits, and the endpoints refuse a conversation the bot is
    not a participant of.

WHAT IS LEFT
------------
`BotApiClient` (transport, retries, one credential), the two fetchers, and the
responder. Nothing in this package can reach a database.
"""

from .client import (
    BotApiClient,
    PlatformAPIError,
    PlatformAuthError,
    PlatformTransientError,
)
from .fetchers import ApiMentionFetcher, ApiMessageFetcher
from .responder import HttpResponder, build_comment_body, build_send_body

__all__ = [
    "ApiMentionFetcher",
    "ApiMessageFetcher",
    "BotApiClient",
    "HttpResponder",
    "PlatformAPIError",
    "PlatformAuthError",
    "PlatformTransientError",
    "build_comment_body",
    "build_send_body",
]
