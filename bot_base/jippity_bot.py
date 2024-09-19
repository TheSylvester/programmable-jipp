from typing import Any, Callable, List, Optional
from dataclasses import dataclass
from nextcord import Message, Embed
import nextcord
from nextcord.ext import commands
from jippity_ai.jippity_core import Jippity
from bot_base.message_chunker import (
    chunk_message_md_friendly,
    get_full_text_from_message,
    send_chunked_message,
)
from bot_base.tool_manager import ToolManager
from jipp.models.jipp_models import LLMError
from jipp.utils.logging_utils import log
from nextcord.errors import NotFound
import asyncio


@dataclass
class Route:
    name: str
    description: str
    condition: Callable[[Message], bool]
    function: Callable[["JippityBot", Message, Callable[[str], Any]], None]


@dataclass
class Command:
    name: str
    description: str
    function: Callable[..., None]
    roles: Optional[List[str]] = None  # Optional list of required roles
    permissions: Optional[List[str]] = None  # Optional list of required permissions


class JippityBot(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.log = log

        # Load AI
        self.jippity = Jippity()

        # Load tools
        self.tool_manager: ToolManager = self.bot.get_cog("ToolManager")
        self.routes: List[Route] = []
        self.custom_commands: List[Command] = []

        # # initial commands
        # self.add_route(
        #     Route(
        #         name="Mention Response",
        #         description="Responds with a greeting when the bot is mentioned",
        #         condition=lambda message: self.bot.user in message.mentions,
        #         function=lambda message, reply_function: self.respond_to_mention(
        #             message, reply_function
        #         ),
        #     )
        # )

    @commands.Cog.listener()
    async def on_ready(self):
        self.log.info(f"JippityBot on_ready username is {self.bot.user.name}")
        self.jippity.set_bot_name(self.bot.user.name)

    async def respond_to_mention(
        self, message: Message, reply_function: Callable[[str], Any]
    ):
        """Responds to @mentions
        Generates basic completion"""
        await self.jippity.chat_response(message, reply_function)

    @commands.Cog.listener()
    async def on_message(self, message: Message):
        if message.author == self.bot.user:
            return

        # Use Nextcord's get_context to see if it's a command
        ctx = await self.bot.get_context(message)
        if ctx.valid:
            await self.bot.process_commands(message)
            return  # Skip custom logic if the message was a valid command

        # Custom logic for non-command messages
        content = await get_full_text_from_message(message)
        channel_history_str = await get_channel_history(message.channel)

        async def send_response(response):
            return await send_chunked_message(message.channel.send, response)

        await self.jippity.message_listener(
            message=content,
            channel_history=channel_history_str,
            send_response=send_response,
        )

    def add_route(self, route: Route):
        self.routes.append(route)

    def add_command(self, command: Command):
        self.custom_commands.append(command)

        @commands.command(name=command.name, brief=command.description)
        async def dynamic_command(ctx, *args, **kwargs):
            # Check if the user has the required roles (if any)
            if command.roles:
                if not any(role.name in command.roles for role in ctx.author.roles):
                    await ctx.send(
                        "You do not have the required role to run this command."
                    )
                    return

            # Check if the user has the required permissions (if any)
            if command.permissions:
                for permission in command.permissions:
                    if not getattr(ctx.author.guild_permissions, permission, False):
                        await ctx.send(f"You lack the permission: {permission}.")
                        return

            await command.function(ctx, *args, **kwargs)

        self.bot.add_command(dynamic_command)

    @commands.command(name="system", brief="Update the system prompt")
    # @commands.has_permissions(administrator=True)
    async def update_system_prompt(self, ctx, *, new_prompt: str):
        result = self.jippity.update_system_prompt(new_prompt)
        await send_chunked_message(ctx.send, result)

    @commands.command(name="model", brief="Update the model")
    # @commands.has_permissions(administrator=True)
    async def update_model(self, ctx, new_model: str = ""):
        result = self.jippity.update_model(new_model)
        await send_chunked_message(ctx.send, result)

    @commands.command(name="clear", brief="Clear the context window")
    # @commands.has_permissions(administrator=True)
    async def clear_context(self, ctx):
        result = self.jippity.clear_context_window()
        await send_chunked_message(ctx.send, result)

    @commands.command(name="models", brief="List available models")
    # @commands.has_permissions(administrator=True)
    async def list_models(self, ctx):
        result = self.jippity.list_models()
        await send_chunked_message(ctx.send, result)

    @commands.command(
        name="commands",
        aliases=["list_commands", "list"],
        brief="List available commands (!list) / tasks (!list tasks)",
    )
    async def list_commands(self, ctx, arg=""):

        first_word = arg.strip().split(" ")[0]
        if first_word == "tasks":
            task_manager = self.bot.get_cog("TaskManager")
            await task_manager.list_tasks(ctx)
            return

        if first_word == "models":
            await self.list_models(ctx)
            return

        if not arg:
            command_list = [command.name for command in self.bot.commands] + [
                command.name for command in self.custom_commands
            ]
            await send_chunked_message(
                ctx.send, f"Available commands: {', '.join(command_list)}"
            )
            return

    @commands.command(name="clear_channel", brief="Clear all messages in the channel")
    async def clear_channel(self, ctx):
        """Clears all messages in the current channel."""
        channel_name = ctx.channel.name
        self.log.info(f"Clearing channel {channel_name}")

        total_deleted = 0
        try:
            while True:
                deleted = await ctx.channel.purge(
                    limit=100
                )  # Delete in smaller batches
                if not deleted:
                    break
                total_deleted += len(deleted)
                await asyncio.sleep(1)  # Short delay between batches

            self.log.info(
                f"Cleared {total_deleted} messages from channel {channel_name}"
            )
            await ctx.send(f"Cleared {total_deleted} messages.", delete_after=5)
        except NotFound as e:
            self.log.warning(f"Error while clearing channel {channel_name}: {str(e)}")
            await ctx.send(
                f"Cleared {total_deleted} messages. Some messages couldn't be deleted.",
                delete_after=5,
            )
        except Exception as e:
            self.log.error(
                f"Unexpected error while clearing channel {channel_name}: {str(e)}"
            )
            await ctx.send(
                f"An error occurred after clearing {total_deleted} messages.",
                delete_after=5,
            )

    @commands.command(name="ask_llms", brief="Ask multiple LLMs a question")
    async def ask_multiple_llms(self, ctx, *, models_prompt: str):
        default_models = [
            "gpt-4o-mini",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "claude-haiku",
        ]
        models: List[str] = []
        prompt = (await get_full_text_from_message(ctx.message)).strip()

        # Parse models and prompt
        words = models_prompt.split()
        model_names = self.jippity.get_model_names()
        for word in words:
            clean_word = word.strip().rstrip(",")
            if clean_word in model_names:
                models.append(clean_word)
            else:
                break

        models = models or default_models
        prompt = " ".join(words[len(models) :]).strip()

        try:
            results = await self.jippity.ask_multiple_llms(models=models, prompt=prompt)

            for model, conversation in results.items():
                if isinstance(conversation, LLMError):
                    await ctx.send(f"Error from {model}: {conversation}")
                    continue

                chunks = chunk_message_md_friendly(str(conversation))

                for i, chunk in enumerate(chunks):
                    embed = Embed(
                        title=f"Response from {model} (Part {i + 1}/{len(chunks)})",
                        description=chunk,
                        color=0x00FF00,
                    )

                    if hasattr(conversation, "usage"):
                        embed.add_field(
                            name="Tokens Used",
                            value=f"{conversation.usage.total_tokens}",
                            inline=False,
                        )

                    await ctx.send(embed=embed)

        except Exception as e:
            await ctx.send(f"An error occurred: {str(e)}")

    @commands.command(name="list_prompts", brief="List available prompts")
    async def list_prompts(self, ctx):
        prompts = self.jippity.list_prompts()
        prompt_list = "\n".join([f"- {prompt}" for prompt in prompts])
        await send_chunked_message(ctx.send, f"Available prompts:\n{prompt_list}")

    @commands.command(name="show_prompt", brief="Show content of a specific prompt")
    async def show_prompt(self, ctx, prompt_name: str):
        try:
            prompt_content = self.jippity.get_prompt(prompt_name)
            await send_chunked_message(
                ctx.send, f"Prompt: {prompt_name}\nContent:\n{prompt_content}"
            )
        except AttributeError:
            await ctx.send(f"Prompt '{prompt_name}' not found.")

    @commands.command(name="ask_llms", brief="Ask multiple LLMs a question")
    async def ask_multiple_llms(self, ctx, *, models_prompt: str):
        default_models = [
            "gpt-4o-mini",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "claude-haiku",
        ]
        models: List[str] = []
        prompt = (await get_full_text_from_message(ctx.message)).strip()

        # Parse models and prompt
        words = models_prompt.split()
        model_names = self.jippity.get_model_names()
        for word in words:
            clean_word = word.strip().rstrip(",")
            if clean_word in model_names:
                models.append(clean_word)
            else:
                break

        models = models or default_models
        prompt = " ".join(words[len(models) :]).strip()

        try:
            results = await self.jippity.ask_multiple_llms(models=models, prompt=prompt)

            for model, conversation in results.items():
                if isinstance(conversation, LLMError):
                    await ctx.send(f"Error from {model}: {conversation}")
                    continue

                chunks = chunk_message_md_friendly(str(conversation))

                for i, chunk in enumerate(chunks):
                    embed = Embed(
                        title=f"Response from {model} (Part {i + 1}/{len(chunks)})",
                        description=chunk,
                        color=0x00FF00,
                    )

                    if hasattr(conversation, "usage"):
                        embed.add_field(
                            name="Tokens Used",
                            value=f"{conversation.usage.total_tokens}",
                            inline=False,
                        )

                    await ctx.send(embed=embed)

        except Exception as e:
            await ctx.send(f"An error occurred: {str(e)}")


async def get_channel_history(channel: nextcord.TextChannel, limit: int = 30) -> str:
    channel_history = await channel.history(limit=limit).flatten()
    channel_history_str = ""
    for channel_message in reversed(channel_history):
        timestamp = channel_message.created_at.strftime("%Y-%m-%d %H:%M:%S")
        author_id = channel_message.author.id
        channel_history_str += f"[{timestamp}] {channel_message.author.name} (ID: {author_id}): {channel_message.content}\n"
    log.debug(f"channel_history_str: {channel_history_str}")
    return channel_history_str


def setup(bot):
    bot.add_cog(JippityBot(bot))
