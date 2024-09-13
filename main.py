import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from dotenv import load_dotenv
import os
from nextcord_bot import NextcordBot

# Load environment variables
load_dotenv()

# Get Discord token from environment variables
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CLIENT_ID = os.getenv("DISCORD_CLIENT_ID")
REDIRECT_URI = os.getenv("REDIRECT_URI")


# Initialize the NextcordBot
bot = NextcordBot()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application is starting up...")
    # Start the Discord bot
    asyncio.create_task(bot.start(DISCORD_TOKEN))
    yield
    print("Application is shutting down...")
    # Close the Discord bot connection
    await bot.close()


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_root():
    return {
        "status": "Application is running",
        "discord_bot": (
            f"Connected as {bot.user}" if bot.is_ready() else "Not connected"
        ),
    }


def generate_oauth2_url():
    scopes = "identify%20guilds%20bot"  # Customize the scopes as needed
    permissions = 8  # Set permissions for the bot (admin in this case)
    return (
        f"https://discord.com/api/oauth2/authorize?"
        f"client_id={DISCORD_CLIENT_ID}&redirect_uri={REDIRECT_URI}&response_type=code&scope={scopes}&permissions={permissions}"
    )


@app.get("/oauth2")
async def oauth2_redirect():
    oauth2_url = generate_oauth2_url()
    return Response(status_code=302, headers={"Location": oauth2_url})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
