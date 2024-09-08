import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
import os
from nextcord_bot import NextcordBot

# Load environment variables
load_dotenv()

# Get Discord token from environment variables
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
