import logging
import os
from dotenv import load_dotenv
import nextcord
from nextcord.ext import commands

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
    intents = nextcord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.messages = True
    return intents


class NextcordBot(commands.Bot):
    def __init__(self):
        intents = setup_intents()
        setup_nextcord_logger()
        super().__init__(command_prefix=COMMAND_PREFIX, intents=intents)
        self.load_extension("jippity_bot")
        self.load_extension("task_manager")

    async def on_ready(self):
        print(f"We have logged in as {self.user}")


if __name__ == "__main__":
    bot = NextcordBot()
    bot.run(DISCORD_TOKEN)
