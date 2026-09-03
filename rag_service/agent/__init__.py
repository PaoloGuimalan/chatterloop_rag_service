"""Function-calling: what the reply generator is allowed to DO, not just say.

Generic on purpose - nothing here knows about chatterloop, Milvus, or
developer_service. `chatterloop.replies.OpenAIReplyGenerator` is the only
current consumer, but the tool contract (config.ToolConfig in, an OpenAI
function schema and an HTTP call out) has no dependency on who is asking.
"""

from .tools import call_tool, to_openai_tool

__all__ = ["call_tool", "to_openai_tool"]
