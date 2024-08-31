from typing import Optional, List, Any
from datetime import datetime


class MessageContext:
    def __init__(
        self,
        content: str,
        author: str,
        channel: str,
        guild: Optional[str] = None,
        attachments: Optional[List[str]] = None,
        mentions: Optional[List[str]] = None,
        reference: Optional[Any] = None,
        is_reply: bool = False,
        created_at: Optional[datetime] = None,
        raw_message: Any = None,
    ):
        self.content = content
        self.author = author
        self.channel = channel
        self.guild = guild
        self.attachments = attachments or []
        self.mentions = mentions or []
        self.reference = reference
        self.is_reply = is_reply
        self.created_at = created_at or datetime.now()
        self.raw_message = raw_message  # Store the original message object if needed

    def __str__(self):
        return (
            f"MessageContext(content='{self.content[:20]}...', "
            f"author='{self.author}', channel='{self.channel}', "
            f"guild='{self.guild}', is_reply={self.is_reply})"
        )

    def to_dict(self):
        return {
            "content": self.content,
            "author": self.author,
            "channel": self.channel,
            "guild": self.guild,
            "attachments": self.attachments,
            "mentions": self.mentions,
            "reference": str(self.reference) if self.reference else None,
            "is_reply": self.is_reply,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_discord_message(cls, message):
        return cls(
            content=message.content,
            author=str(message.author),
            channel=str(message.channel),
            guild=str(message.guild) if message.guild else None,
            attachments=[a.filename for a in message.attachments],
            mentions=[str(m) for m in message.mentions],
            reference=(
                message.reference.resolved.content
                if message.reference and message.reference.resolved
                else None
            ),
            is_reply=bool(message.reference),
            created_at=message.created_at,
            raw_message=message,
        )
