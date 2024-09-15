from nextcord import TextChannel


async def get_channel_history(channel: TextChannel, limit: int = 30) -> str:
    channel_history = await channel.history(limit=limit).flatten()
    channel_history_str = ""
    for channel_message in reversed(channel_history):
        timestamp = channel_message.created_at.strftime("%Y-%m-%d %H:%M:%S")
        channel_history_str += (
            f"[{timestamp}] {channel_message.author}: {channel_message.content}\n"
        )
    return channel_history_str
