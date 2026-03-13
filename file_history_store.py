import json
import os
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict


def get_history(session_id):
    return FileChatMessageHistory(session_id, "./chat_history")


class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        self.session_id = session_id
        self.storage_path = storage_path
        self.file_path = os.path.join(storage_path, f"{session_id}.json")
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self._cached_messages = None  # 缓存消息，避免重复读文件

    @property
    def messages(self) -> list[BaseMessage]:
        if self._cached_messages is not None:
            return self._cached_messages
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)
                self._cached_messages = messages_from_dict(messages_data)
                return self._cached_messages
        except FileNotFoundError:
            self._cached_messages = []
            return []

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        all_messages = list(self.messages)
        all_messages.extend(messages)
        new_messages = [message_to_dict(m) for m in all_messages]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f)
        self._cached_messages = all_messages  # 更新缓存

    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)
