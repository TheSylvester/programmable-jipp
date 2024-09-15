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

        # Load AI
        self.jippity = Jippity()
        self.jippity.load_prompts("jippity_ai/prompts")

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
        log.info(f"JippityBot on_ready username is {self.bot.user.name}")
        self.jippity.set_bot_name(self.bot.user.name)

    async def respond_to_mention(
        self, message: Message, reply_function: Callable[[str], Any]
    ):
        """Responds to @mentions
        Generates basic completion"""
        await self.jippity.chat_response(message, reply_function)

    # @commands.Cog.listener()
    # async def on_message(self, message: Message):
    #     if message.author == self.bot.user:
    #         return

    #     async def respond_fn(response):
    #         return await send_chunked_message(message.channel.send, response)

    #     for route in self.routes:
    #         if route.condition(message):
    #             await route.function(
    #                 message,
    #                 respond_fn,
    #             )
    #             return

    @commands.Cog.listener()
    async def on_message(self, message: Message):
        log.debug(f"on_message: {message.content}")
        if message.author == self.bot.user:
            return

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

        # if the first words in models_prompt are models, then they are models
        words = models_prompt.split()
        model_names = self.jippity.get_model_names()
        for word in words:
            clean_word = word.strip().rstrip(",")
            if clean_word in model_names:
                models.append(clean_word)
            else:
                break

        # If no valid models found, use default models
        if not models:
            models = default_models

        # Extract the actual prompt
        prompt = " ".join(words[len(models) :]).strip()

        try:
            results = await self.jippity.ask_multiple_llms(models=models, prompt=prompt)

            # Create an embed for each model response, using chunked text if necessary
            for model, conversation in results.items():
                # If conversation is an error, handle it as an error
                if isinstance(conversation, LLMError):
                    await ctx.send(f"Error from {model}: {conversation}")
                    continue

                # Chunk the conversation text before adding to the embed
                chunks = chunk_message_md_friendly(str(conversation))

                # Create embeds for each chunk of conversation
                for i, chunk in enumerate(chunks):
                    embed = Embed(
                        title=f"Response from {model} (Part {i + 1}/{len(chunks)})",
                        description=chunk,
                        color=0x00FF00,  # Optional: Set a color for the embed
                    )

                    # Optionally, add token usage stats if available
                    if hasattr(conversation, "usage"):
                        embed.add_field(
                            name="Tokens Used",
                            value=f"{conversation.usage.total_tokens}",
                            inline=False,
                        )

                    # Send each embed chunk
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

        # if the first words in models_prompt are models, then they are models
        words = models_prompt.split()
        model_names = self.jippity.get_model_names()
        for word in words:
            clean_word = word.strip().rstrip(",")
            if clean_word in model_names:
                models.append(clean_word)
            else:
                break

        # If no valid models found, use default models
        if not models:
            models = default_models

        # Extract the actual prompt
        prompt = " ".join(words[len(models) :]).strip()

        try:
            results = await self.jippity.ask_multiple_llms(models=models, prompt=prompt)

            # Create an embed for each model response, using chunked text if necessary
            for model, conversation in results.items():
                # If conversation is an error, handle it as an error
                if isinstance(conversation, LLMError):
                    await ctx.send(f"Error from {model}: {conversation}")
                    continue

                # Chunk the conversation text before adding to the embed
                chunks = chunk_message_md_friendly(str(conversation))

                # Create embeds for each chunk of conversation
                for i, chunk in enumerate(chunks):
                    embed = Embed(
                        title=f"Response from {model} (Part {i + 1}/{len(chunks)})",
                        description=chunk,
                        color=0x00FF00,  # Optional: Set a color for the embed
                    )

                    # Optionally, add token usage stats if available
                    if hasattr(conversation, "usage"):
                        embed.add_field(
                            name="Tokens Used",
                            value=f"{conversation.usage.total_tokens}",
                            inline=False,
                        )

                    # Send each embed chunk
                    await ctx.send(embed=embed)

        except Exception as e:
            await ctx.send(f"An error occurred: {str(e)}")


async def get_channel_history(channel: nextcord.TextChannel, limit: int = 30) -> str:
    channel_history = await channel.history(limit=limit).flatten()
    channel_history_str = ""
    for channel_message in reversed(channel_history):
        timestamp = channel_message.created_at.strftime("%Y-%m-%d %H:%M:%S")
        channel_history_str += (
            f"[{timestamp}] {channel_message.author}: {channel_message.content}\n"
        )
    log.debug(f"channel_history_str: {channel_history_str}")
    return channel_history_str


def setup(bot):
    bot.add_cog(JippityBot(bot))
