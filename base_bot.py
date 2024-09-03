import os
from dotenv import load_dotenv
import nextcord
from nextcord.ext import commands
from jippity_core.jippity import Jippity
from models.message_context import MessageContext

# Load environment variables
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
COMMAND_PREFIX = "!"


def setup_intents():
    intents = nextcord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.messages = True
    return intents


class DiscordBot(commands.Bot):
    def __init__(self):
        intents = setup_intents()
        super().__init__(command_prefix=COMMAND_PREFIX, intents=intents)
        self.jippity = Jippity()

    async def on_ready(self):
        print(f"We have logged in as {self.user}")

    async def on_message(self, message: nextcord.Message):
        if message.author == self.user:
            return

        await self.process_commands(message)

        if message.content.startswith(COMMAND_PREFIX):
            return

        context = MessageContext.from_discord_message(message)
        await self.jippity.handle_message(
            context=context, send_response=message.channel.send
        )


@commands.command(name="system")
# @commands.has_permissions(administrator=True)
async def update_system_prompt(ctx, *, new_prompt: str):
    result = bot.jippity.update_system_prompt(new_prompt)
    await ctx.send(result)


@commands.command(name="model")
# @commands.has_permissions(administrator=True)
async def update_model(ctx, new_model: str = ""):
    result = bot.jippity.update_model(new_model)
    await ctx.send(result)


@commands.command(name="clear")
# @commands.has_permissions(administrator=True)
async def clear_context(ctx):
    result = bot.jippity.clear_context_window()
    await ctx.send(result)


@commands.command(name="list_models")
# @commands.has_permissions(administrator=True)
async def list_models(ctx):
    result = bot.jippity.list_models()
    await ctx.send(result)


@commands.command(name="commands", aliases=["list_commands"])
# @commands.has_permissions(administrator=True)
async def list_commands(ctx):
    command_list = [command.name for command in bot.commands]
    await ctx.send(f"Available commands: {', '.join(command_list)}")


# Now create the bot instance after defining all commands
bot = DiscordBot()

# Add these commands to the bot
bot.add_command(update_system_prompt)
bot.add_command(update_model)
bot.add_command(clear_context)
bot.add_command(list_models)
bot.add_command(list_commands)

# ... rest of the file ...

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
