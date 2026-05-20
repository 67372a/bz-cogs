from dataclasses import dataclass, field
from typing import Literal, Optional, Union


@dataclass(frozen=True)
class MessageEntry:
    role: Literal['user', 'assistant', 'system', 'tool']
    content: Union[str, list]
    name: str = None
    tool_calls: list = field(default_factory=list)
    tool_call_id: Optional[str] = None
    reasoning_details: list = None
