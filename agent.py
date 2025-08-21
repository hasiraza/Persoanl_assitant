from dotenv import load_dotenv
import os

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    noise_cancellation,
    openai,
    silero,  # For VAD (Voice Activity Detection)
)
from prompt import AGENT_INSTRUCTION, SESSION_INSTRUCTION
from tool import get_weather, search_web, send_email

load_dotenv(".env", override=True)  # Force override existing env vars

# Debug: Check what key was loaded
openai_key = os.getenv('OPENAI_API_KEY')
if openai_key:
    print(f"✓ OpenAI API Key loaded: {openai_key[:15]}...{openai_key[-4:]}")
    print(f"✓ Key length: {len(openai_key)} characters")
    if openai_key.startswith('sk-proj'):
        print("✓ Using project-scoped API key")
    elif openai_key.startswith('sk-'):
        print("✓ Using user-scoped API key")
    else:
        print("✗ Invalid API key format!")
else:
    print("✗ No OpenAI API Key found!")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=AGENT_INSTRUCTION,
            # Uncomment when you want to add tools
            tools=[
                get_weather,
                search_web,
                send_email
            ],
        )


async def entrypoint(ctx: agents.JobContext):
    # Connect to the room first
    await ctx.connect()
    
    # Create the session with all required components
    session = AgentSession(
        # Using OpenAI Realtime API for everything
        llm=openai.realtime.RealtimeModel(
            voice="coral",
        ),
        vad=silero.VAD.load(),
    )

    # Start the session
    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            video_enabled=True,
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Generate the initial reply
    await session.generate_reply(
        instructions=SESSION_INSTRUCTION,
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))