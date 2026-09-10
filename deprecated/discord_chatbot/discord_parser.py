import json
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DiscordMessage:
    content: str
    author: str
    timestamp: str
    channel: str
    channel_id: str
    message_id: str
    guild_id: str
    
    def to_document(self) -> Dict[str, Any]:
        return {
            "page_content": self.content,
            "metadata": {
                "author": self.author,
                "timestamp": self.timestamp,
                "channel": self.channel,
                "channel_id": self.channel_id,
                "message_id": self.message_id,
                "guild_id": self.guild_id,
                "source": "discord"
            }
        }


class DiscordChatParser:
    def __init__(self, json_path: str):
        self.json_path = json_path
        
    def parse(self) -> List[DiscordMessage]:
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        messages = []
        channel_name = data.get("channel", {}).get("name", "unknown")
        channel_id = data.get("channel", {}).get("id", "unknown")
        guild_id = data.get("guild", {}).get("id", "unknown")
        
        for msg in data.get("messages", []):
            # Skip messages without content
            if not msg.get("content"):
                continue
                
            message = DiscordMessage(
                content=msg["content"],
                author=msg.get("author", {}).get("name", "unknown"),
                timestamp=msg.get("timestamp", ""),
                channel=channel_name,
                channel_id=channel_id,
                message_id=msg.get("id", ""),
                guild_id=guild_id
            )
            messages.append(message)
            
        return messages
    
    def get_documents(self) -> List[Dict[str, Any]]:
        messages = self.parse()
        return [msg.to_document() for msg in messages]