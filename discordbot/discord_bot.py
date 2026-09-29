import logging
import asyncio
import os
import discord
from discord.ext import commands
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# --- CONFIG ---
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "").strip()
_channel_raw = os.getenv("DISCORD_CHANNEL_ID", "").strip()
if not _channel_raw.isdigit():
    raise RuntimeError("DISCORD_CHANNEL_ID must be set to a numeric channel id (no default committed)")
CHANNEL_ID = int(_channel_raw)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# Try to parse webhook id from webhook URL (format: /api/webhooks/<id>/<token>)
WEBHOOK_ID = None
try:
    if DISCORD_WEBHOOK_URL:
        import re
        m = re.search(r"/webhooks/(\d+)", DISCORD_WEBHOOK_URL)
        if m:
            WEBHOOK_ID = int(m.group(1))
except Exception:
    WEBHOOK_ID = None

# --- LOGGING ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# --- BOT SETUP ---
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# --- ALERT FUNCTION (call this from your website/FastAPI) ---
async def send_alert(message: str):
    """Send an alert message to the configured Discord channel."""
    channel = bot.get_channel(CHANNEL_ID)
    if channel is None:
        logging.warning("Channel not found. Is CHANNEL_ID correct and bot added to server?")
        return
    try:
        await channel.send(message)
        logging.info("Alert sent to Discord: %s", message[:50])
    except Exception as e:
        logging.error("Failed to send Discord alert: %s", e)

# --- COMMANDS ---
@bot.event
async def on_ready():
    print(f"✅ Bot is online as {bot.user}")
    logging.info("Discord bot ready: %s", bot.user)

@bot.command(name="hello")
async def hello(ctx):
    """!hello — test command"""
    await ctx.send("🌍 Hello, World! Your Discord alert bot is working.")

@bot.command(name="testalert")
async def test_alert(ctx):
    """!testalert — sends a test alert to the alerts channel"""
    await send_alert("🔔 Test alert from SynCura!")
    await ctx.send("✅ Alert sent to the alerts channel!")

@bot.command(name="status")
async def status(ctx):
    """!status — check bot is alive"""
    await ctx.send("🟢 SynCura alert bot is running.")


@bot.command(name="clear")
@commands.has_permissions(manage_messages=True)
async def clear(ctx, amount: int = 25):
    """!clear [amount] — delete the last [amount] messages in this channel.

    Requires the user to have Manage Messages permission. Note: Discord
    prevents bulk deletion of messages older than 14 days; those will be
    skipped automatically by the API.
    """
    if amount < 1 or amount > 100:
        await ctx.send("Please specify an amount between 1 and 100.")
        return
    try:
        deleted = await ctx.channel.purge(limit=amount)
        await ctx.send(f"✅ Deleted {len(deleted)} messages.", delete_after=6)
        logging.info("User %s cleared %d messages in %s", ctx.author, len(deleted), ctx.channel)
    except Exception as e:
        logging.error("Failed to purge messages: %s", e)
        await ctx.send(f"❌ Failed to delete messages: {e}")


@bot.command(name="clearalerts")
@commands.has_permissions(manage_messages=True)
async def clear_alerts(ctx, limit: int = 200):
    """!clearalerts [limit] — remove recent messages sent by the bot in the alerts channel.

    Scans the configured alerts channel and deletes up to [limit] messages authored
    by the bot. This helps keep the alerts channel tidy.
    """
    try:
        channel = bot.get_channel(CHANNEL_ID)
        if channel is None:
            await ctx.send("Alerts channel not found. Is CHANNEL_ID correct?")
            return

        def is_bot_msg(m):
            # Delete messages authored by the bot or posted via the configured webhook
            try:
                if m.author == bot.user:
                    return True
                # Many webhook messages expose `webhook_id` attribute on Message
                wid = getattr(m, 'webhook_id', None)
                if wid is not None and WEBHOOK_ID is not None and int(wid) == WEBHOOK_ID:
                    return True
                # Fallback: match by author name used by webhook
                if DISCORD_WEBHOOK_URL and getattr(m.author, 'name', '').lower().startswith('syncura'):
                    return True
            except Exception:
                return False
            return False

        deleted = await channel.purge(limit=limit, check=is_bot_msg)
        await ctx.send(f"✅ Removed {len(deleted)} bot messages from the alerts channel.", delete_after=6)
        logging.info("Cleared %d bot messages from alerts channel", len(deleted))
    except Exception as e:
        logging.error("Failed to clear alerts: %s", e)
        await ctx.send(f"❌ Failed to clear alerts: {e}")

# --- MAIN ---
if __name__ == "__main__":
    print("Bot is starting...")
    bot.run(DISCORD_TOKEN)
