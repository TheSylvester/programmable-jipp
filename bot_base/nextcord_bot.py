import logging
import os
from dotenv import load_dotenv
import nextcord
from nextcord.ext import commands
from jipp.utils.logging_utils import log
import aiohttp
from pydantic import BaseModel

# Load environment variables
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
NEXTCORD_LOG_FILENAME = "nextcord_bot.log"

COMMAND_PREFIX = "!"


def setup_nextcord_logger():

    # Set up a separate logger for Nextcord
    nextcord_logger = logging.getLogger("nextcord")
    nextcord_logger.setLevel(logging.INFO)

    # Create file handler for Nextcord logs
    nextcord_file_handler = logging.FileHandler(NEXTCORD_LOG_FILENAME)
    nextcord_file_handler.setLevel(logging.INFO)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    nextcord_file_handler.setFormatter(formatter)

    # Add the handler to the Nextcord logger
    nextcord_logger.addHandler(nextcord_file_handler)

    # Disable propagation to prevent logs from being passed to the root logger
    nextcord_logger.propagate = False


def setup_intents():
    """Grants perms to nextcord bot"""
    intents = nextcord.Intents.default()
    intents.message_content = True  # Handle message content
    intents.guilds = True  # Access to guild events (joining/leaving)
    intents.messages = True  # Read messages in guild channels
    intents.members = True  # Access to member-related events (@mentions)
    intents.reactions = True  # Track message reactions
    intents.typing = True  # Handle typing events
    intents.dm_messages = True  # Allow direct messages to bot

    # priviledged
    intents.presences = True

    return intents


class Token(BaseModel):
    access_token: str
    token_type: str


class UserInfo(BaseModel):
    id: str
    username: str
    discriminator: str
    avatar: str | None


class NextcordBot(commands.Bot):
    def __init__(self):
        intents = nextcord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=COMMAND_PREFIX, intents=intents)

        # Load extensions
        self.load_extension("bot_base.tool_manager")
        self.load_extension("bot_base.jippity_bot")

    async def on_ready(self):
        log.info(f"We have logged in as {self.user}")

    async def get_access_token(self, code: str) -> Token:
        async with aiohttp.ClientSession() as session:
            token_url = "https://discord.com/api/oauth2/token"
            data = {
                "client_id": self.application_id,
                "client_secret": self.http.client.client_secret,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": "YOUR_REDIRECT_URI",
            }
            async with session.post(token_url, data=data) as resp:
                if resp.status != 200:
                    raise HTTPException(
                        status_code=400, detail="Failed to retrieve access token"
                    )
                token_data = await resp.json()
                return Token(**token_data)

    async def get_user_info(self, access_token: str) -> UserInfo:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {access_token}"}
            async with session.get(
                "https://discord.com/api/users/@me", headers=headers
            ) as resp:
                if resp.status != 200:
                    raise HTTPException(
                        status_code=400, detail="Failed to retrieve user info"
                    )
                user_data = await resp.json()
                return UserInfo(**user_data)


if __name__ == "__main__":
    bot = NextcordBot()
    bot.run(DISCORD_TOKEN)
