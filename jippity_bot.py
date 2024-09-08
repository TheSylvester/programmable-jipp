import re
from typing import Any, Callable, List, Optional
from dataclasses import dataclass
from nextcord import Message
from nextcord.ext import commands
from jippity.jippity_core import Jippity
from message_chunker import send_chunked_message
from task_manager import CreateTask


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
        self.jippity = Jippity()
        self.routes: List[Route] = []
        self.custom_commands: List[Command] = []

        # initial commands
        self.add_route(
            Route(
                name="Mention Response",
                description="Responds with a greeting when the bot is mentioned",
                condition=lambda message: self.bot.user in message.mentions,
                function=lambda message, reply_function: self.respond_to_mention(
                    message, reply_function
                ),
            )
        )

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

        async def respond_fn(response):
            return await send_chunked_message(message.channel.send, response)

        for route in self.routes:
            if route.condition(message):
                await route.function(
                    message,
                    respond_fn,
                )
                return

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

    @commands.command(name="task", brief="Set a repeating task")
    async def create_task(self, ctx, *, prompt: str):
        task_manager = self.bot.get_cog("TaskManager")

        def create_task(task_name: str, interval: int, prompt: str):
            task_manager.create_task(
                task_name=task_name,
                interval=interval,
                function=lambda _: self.jippity.chat(prompt),
            )

        tools = [{"function": create_task, "schema": CreateTask}]

        response = await self.jippity.ask_llm_with_tools(
            model="llama3-groq-8b-8192-tool-use-preview",
            prompt=prompt,
            tools=tools,
        )
        await send_chunked_message(ctx.send, response)

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
        brief="List available commands",
    )
    async def list_commands(self, ctx):
        command_list = [command.name for command in self.bot.commands] + [
            command.name for command in self.custom_commands
        ]
        await send_chunked_message(
            ctx.send, f"Available commands: {', '.join(command_list)}"
        )


def setup(bot):
    bot.add_cog(JippityBot(bot))
