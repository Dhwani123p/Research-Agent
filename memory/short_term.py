"""Short-term memory — holds the current research session's gathered facts."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ResearchNote:
    source: str
    content: str
    tool_used: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ShortTermMemory:
    """
    Stores notes gathered during the current research session.
    Injected into the LLM context window as a running knowledge base.
    """

    def __init__(self):
        self.notes: list[ResearchNote] = []
        self.query: str = ""
        self.sub_tasks: list[str] = []
        self.completed_tasks: list[str] = []

    def set_query(self, query: str):
        self.query = query

    def set_sub_tasks(self, tasks: list[str]):
        self.sub_tasks = tasks

    def mark_task_done(self, task: str):
        if task not in self.completed_tasks:
            self.completed_tasks.append(task)

    def add_note(self, source: str, content: str, tool_used: str):
        self.notes.append(ResearchNote(source=source, content=content, tool_used=tool_used))

    def get_context_block(self) -> str:
        """Format all notes into a context string for the LLM."""
        if not self.notes:
            return "No notes gathered yet."
        lines = []
        for i, note in enumerate(self.notes, 1):
            lines.append(f"[Note {i} | Tool: {note.tool_used} | Source: {note.source}]")
            lines.append(note.content)
            lines.append("")
        return "\n".join(lines)

    def pending_tasks(self) -> list[str]:
        return [t for t in self.sub_tasks if t not in self.completed_tasks]

    def clear(self):
        self.__init__()
