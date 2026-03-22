from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore


def get_checkpointer() -> MemorySaver:
    return MemorySaver()


def get_store() -> InMemoryStore:
    return InMemoryStore()
