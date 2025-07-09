import discord
from discord.ext import commands               # ⬅ classic commands helpers
import os
from dotenv import load_dotenv
from rag_chatbot import DiscordRAGChatbot
load_dotenv()

# Initialize the bot with all intents
bot = discord.Bot(intents=discord.Intents.all())

# Initialize the RAG chatbot
rag_chatbot = None

@bot.event
async def on_ready():
    global rag_chatbot
    print(f"{bot.user} has connected to Discord!")
    
    # Initialize RAG chatbot
    try:
        rag_chatbot = DiscordRAGChatbot(
            vector_store_path=os.getenv("VECTOR_STORE_PATH", "./discord_vector_store")
        )
        print("RAG chatbot initialized successfully!")
    except Exception as e:
        print(f"Error initializing RAG chatbot: {e}")
@bot.event
async def on_disconnect():
    print("⬇️ Disconnected!")

@commands.cooldown(1, 60, commands.BucketType.user)
@bot.slash_command(name="ask", description="Ask a question about the Discord chat history")
async def ask(ctx: discord.ApplicationContext, *, question: str):
    # Defer the response as it might take a few seconds
    await ctx.defer()
    
    if not rag_chatbot:
        await ctx.followup.send("❌ RAG chatbot is not initialized. Please check the logs.")
        return
    
    try:
        # Query the RAG system first
        result = rag_chatbot.query(question, k=5)
        
        # Create the main embed with the answer
        embed = discord.Embed(
            title="Answer",
            description=result['answer'][:4096],  # Discord embed description limit
            color=discord.Color.blue()
        )
        
        # Add footer with token usage
        embed.set_footer(text=f"Tokens used: {result['usage']['total_tokens']} | Model: {result['model']}")
        
        # Add citation fields (limit to 5 for Discord's field limit)
        for i, source in enumerate(result['sources'][:5], start=1):
            # Get Discord IDs for creating jump link
            message_id = source.get('message_id', 'unknown')
            channel_id = source.get('channel_id', 'unknown')
            guild_id = source.get('guild_id', 'unknown')
            
            # Format timestamp nicely
            timestamp = source['timestamp'][:10] if source['timestamp'] else 'Unknown date'
            
            # Create field for this citation
            field_name = f"[{i}] {source['author']} – {timestamp}"
            
            # Truncate content to fit Discord's field value limit (1024 chars)
            content_preview = source['content'][:150] + "..." if len(source['content']) > 150 else source['content']
            
            # Create jump link if we have all required IDs
            if message_id != 'unknown' and channel_id != 'unknown' and guild_id != 'unknown':
                jump_link = f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"
                field_value = f"[Jump to message]({jump_link})\nScore: {source['score']:.3f}\n{content_preview}"
            else:
                field_value = f"Score: {source['score']:.3f}\n{content_preview}"
            
            embed.add_field(
                name=field_name,
                value=field_value,
                inline=False
            )
        
        # Check if we can create threads (must be in a guild and in a text channel)
        if ctx.guild and hasattr(ctx.channel, 'create_thread'):
            # Send initial message
            initial_msg = await ctx.followup.send(f"📚 **Question:** {question[:100]}{'...' if len(question) > 100 else ''}")
            
            # Get the actual message object from the channel
            try:
                # Fetch the message we just sent
                message = await ctx.channel.fetch_message(initial_msg.id)
                
                # Create a thread from the message
                thread = await message.create_thread(
                    name=f"Q: {question[:50]}{'...' if len(question) > 50 else ''}",
                    auto_archive_duration=60
                )
                
                # Send the answer in the thread
                await thread.send(embed=embed)
            except Exception as e:
                # If thread creation fails, just send the embed as a followup
                print(f"Thread creation failed: {e}")
                await ctx.followup.send(embed=embed)
        else:
            # In DMs or channels without thread support, send normally
            await ctx.followup.send(embed=embed)
        
    except Exception as e:
        error_embed = discord.Embed(
            title="Error",
            description=f"An error occurred while processing your question: {str(e)}",
            color=discord.Color.red()
        )
        await ctx.followup.send(embed=error_embed)

# Add a help command
@bot.slash_command(name="rag_help", description="Get help with the RAG chatbot")
async def rag_help(ctx: discord.ApplicationContext):
    help_embed = discord.Embed(
        title="Discord RAG Chatbot Help",
        description="Ask questions about the Discord chat history using AI-powered search and generation.",
        color=discord.Color.green()
    )
    
    help_embed.add_field(
        name="How to use",
        value="Use `/ask` followed by your question to search through the Discord history and get an AI-generated answer. The bot will create a thread for the response to keep channels organized.",
        inline=False
    )
    
    help_embed.add_field(
        name="Example",
        value="`/ask What is the Vesuvius Challenge?`",
        inline=False
    )
    
    help_embed.add_field(
        name="Features",
        value="• Searches through Discord chat history\n• Uses AI to generate contextual answers\n• Creates threads for responses to avoid clutter\n• Shows sources with relevance scores\n• Powered by GPT-4.0 mini",
        inline=False
    )
    
    await ctx.respond(embed=help_embed)
@ask.error
async def ask_error(ctx, exc):
    if isinstance(exc, commands.CommandOnCooldown):
        await ctx.respond(
            f"Whoa — try again in {exc.retry_after:.1f}s.",
            ephemeral=True,
        )

# Run the bot
if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        print("Error: DISCORD_TOKEN not found in environment variables!")
        print("Please add DISCORD_TOKEN to your .env file")
    else:
        bot.run(token)