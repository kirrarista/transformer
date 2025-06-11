import os
import aiohttp
from twitchio.ext import commands
from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import TextClassificationPipeline
import torch

class Bot(commands.Bot):
    def __init__(self):
        # Replace the fallback strings with your actual credentials
        super().__init__(
            token="tyour_oken",
            client_id="your_client_id",
            nick="BigBrotherBot",
            prefix="!",
            initial_channels=['your_channel_name'],  # Replace with your channel name
        )
        # Set these appropriately:
        self.broadcaster_id = "1111111111"  # Replace with your Twitch user ID
        self.moderator_id  = "1111111111"  # Replace with your Twitch moderator ID
        self.api_oauth_token = "oauth:your_token"  # Replace with your OAuth token
        self.toxicity_threshold = 0.5 # Default threshold for toxicity

    async def event_ready(self):
        print(f"Bot is ready! Logged in as {self.nick}")

    async def delete_message_api(self, message_id: str):
        """
        Calls the Twitch API to delete a message given its message_id.
        The URL requires broadcaster_id, moderator_id, and message_id as query parameters.
        """
        url = (
            f"https://api.twitch.tv/helix/moderation/chat?"
            f"broadcaster_id={self.broadcaster_id}"
            f"&moderator_id={self.moderator_id}"
            f"&message_id={message_id}"
        )
        headers = {
            "Authorization": "Bearer your_token",
            "Client-Id": "your_client_id",
        }
        async with aiohttp.ClientSession() as session:
            async with session.delete(url, headers=headers) as resp:
                if not resp.status == 204:
                    error_text = await resp.text()
                    print(f"Failed to delete message {message_id} via API: {resp.status} {error_text}")

    async def event_message(self, message):
        # Check for toxicity in the message content
        if evaluate(message.content, self.toxicity_threshold):
            # Extract the message ID from tags (ensure your IRC connection includes the tag capability)
            message_id = message.tags.get("id") if message.tags else None
            if message_id and not (message.echo or message.author.name.lower() == self.nick.lower()):
                await self.delete_message_api(message_id)
            else:
                print("Message did not include a valid 'id' tag; cannot delete.")
            # Stop further processing.
            return

        # Process any commands attached to the message
        await self.handle_commands(message)
    
    # Command to set the toxicity threshold
    @commands.command(name="setthreshold")
    async def set_threshold(self, ctx: commands.Context):
        if not ctx.author.is_mod:
            await ctx.send(f"@{ctx.author.name}, only moderators can change the threshold.")
            return
    
        try:
            new_threshold = float(ctx.message.content.split(" ", 1)[1])
            if 0 <= new_threshold <= 1:
                self.toxicity_threshold = new_threshold
                await ctx.send(f"Toxicity threshold updated to {new_threshold:.2f}")
            else:
                await ctx.send("Please enter a value between 0 and 1.")
        except (IndexError, ValueError):
            await ctx.send("Usage: !setthreshold 0.65")

def evaluate(raw: str, toxicity_threshold: float = 0.75) -> bool:    
    clean = ''.join(c for c in raw if c.isalnum() or c == ' ').strip().lower()

    result = pipe(clean)
    toxic_prob = result[0][1]["score"]

    print(f"Text: {clean}")
    print(f"Toxic Probability: {toxic_prob:.4f}")
    return toxic_prob > toxicity_threshold

if __name__ == '__main__':
    model = AutoPeftModelForSequenceClassification.from_pretrained("lora-toxic-roberta")
    tokenizer = AutoTokenizer.from_pretrained("lora-toxic-roberta")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    model.to(device)

    pipe = TextClassificationPipeline(
        model=model.base_model.model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=0
    )

    bot = Bot()
    bot.run()