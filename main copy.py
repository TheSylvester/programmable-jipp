import asyncio
from contextlib import asynccontextmanager
import uuid
from fastapi import FastAPI, Response, Request, HTTPException, status
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from dotenv import load_dotenv
import os
from bot_base.nextcord_bot import NextcordBot
from jipp.utils.logging_utils import log
import httpx
import urllib.parse

# Load environment variables
load_dotenv()

# Get Discord credentials from environment variables
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CLIENT_ID = os.getenv("DISCORD_CLIENT_ID")
DISCORD_CLIENT_SECRET = os.getenv("DISCORD_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")

if not all([DISCORD_TOKEN, DISCORD_CLIENT_ID, DISCORD_CLIENT_SECRET, REDIRECT_URI]):
    log.error("One or more required environment variables are missing.")
    raise EnvironmentError("Missing required environment variables.")

# Initialize the NextcordBot
bot = NextcordBot()

# Initialize FastAPI app
app = FastAPI()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Application is starting up...")
    # Start the Discord bot
    asyncio.create_task(bot.start(DISCORD_TOKEN))
    yield
    log.info("Application is shutting down...")
    # Close the Discord bot connection
    await bot.close()


app.router.lifespan_context = lifespan


def generate_oauth2_url(state: str):
    """
    Generates the Discord OAuth2 URL for bot authorization.

    Args:
        state (str): A unique string to maintain state between the request and callback.

    Returns:
        str: The complete OAuth2 URL.
    """
    # Correct scope and URL encoding
    scopes = (
        "identify guilds bot"  # Scopes should be space-separated, not URL-encoded here
    )
    permissions = 8  # Admin permissions
    query = {
        "client_id": DISCORD_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": scopes,  # Don't manually URL-encode this part
        "permissions": permissions,
        "state": state,
    }
    oauth2_url = (
        f"https://discord.com/api/oauth2/authorize?{urllib.parse.urlencode(query)}"
    )
    return oauth2_url


@app.get("/")
async def read_root():
    """
    Root endpoint to check the status of the application and the Discord bot.

    Returns:
        dict: Status information.
    """
    return {
        "status": "Application is running",
        "discord_bot": (
            f"Connected as {bot.user}" if bot.is_ready() else "Not connected"
        ),
    }


@app.get("/oauth2")
async def oauth2_redirect():
    """
    Endpoint to redirect users to the Discord OAuth2 authorization URL.

    Returns:
        RedirectResponse: Redirects the user to Discord's OAuth2 page.
    """
    # Generate a unique state parameter for security (e.g., using UUID in production)
    state = str(uuid.uuid4())
    oauth2_url = generate_oauth2_url(state)
    return RedirectResponse(oauth2_url)


@app.get("/callback")
async def oauth_callback(
    request: Request, code: str = None, state: str = None, error: str = None
):
    """
    Callback endpoint for Discord OAuth2. Handles token exchange and user feedback.

    Args:
        request (Request): The incoming request.
        code (str, optional): The authorization code returned by Discord.
        state (str, optional): The state parameter to verify.
        error (str, optional): Error message if authorization failed.

    Returns:
        HTMLResponse: A response indicating success or failure.
    """
    if error:
        log.error(f"OAuth2 Error: {error}")
        return HTMLResponse(
            f"<h1>Authorization Failed</h1><p>{error}</p>", status_code=400
        )

    if not code:
        log.error("No authorization code provided.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No code provided."
        )

    # Verify state parameter if implemented (omitted for brevity)

    # Exchange the authorization code for an access token
    token_url = "https://discord.com/api/oauth2/token"
    data = {
        "client_id": DISCORD_CLIENT_ID,
        "client_secret": DISCORD_CLIENT_SECRET,
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "scope": "identify guilds bot",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(token_url, data=data, headers=headers)
            response.raise_for_status()
            token_data = response.json()
            log.info(f"Token data received: {token_data}")
        except httpx.HTTPError as e:
            log.error(f"HTTP error during token exchange: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token exchange failed.",
            )

    # Optionally, use the access token to fetch user information
    access_token = token_data.get("access_token")
    if not access_token:
        log.error("No access token received.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No access token received.",
        )

    user_api_url = "https://discord.com/api/users/@me"
    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient() as client:
        try:
            user_response = await client.get(user_api_url, headers=headers)
            user_response.raise_for_status()
            user_data = user_response.json()
            log.info(f"User data fetched: {user_data}")
        except httpx.HTTPError as e:
            log.error(f"HTTP error fetching user data: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch user data.",
            )

    # Here, you can implement storing user data or associating tokens with user sessions
    # For simplicity, we'll just return a success message
    return HTMLResponse(
        f"""
        <h1>Authorization Successful</h1>
        <p>Welcome, {user_data.get('username')}#{user_data.get('discriminator')}!</p>
        <p>Your bot has been added to the server.</p>
    """
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom handler for HTTP exceptions to provide meaningful error messages.

    Args:
        request (Request): The incoming request.
        exc (HTTPException): The exception raised.

    Returns:
        JSONResponse: JSON response with error details.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


if __name__ == "__main__":
    import uvicorn

    log.info("Starting the application...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
