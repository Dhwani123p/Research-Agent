"""Main agent orchestration loop — implements the Planner → Executor → Critic → Synthesizer pattern."""

import json
import os
from typing import Generator
import anthropic

from agent.prompts import (
    PLANNER_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
    CRITIC_PROMPT,
    SYNTHESIZER_PROMPT,
)
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from tools.web_search import web_search, WEB_SEARCH_TOOL
from tools.fetch_page import fetch_page, FETCH_PAGE_TOOL
from tools.arxiv_search import arxiv_search, ARXIV_SEARCH_TOOL
from tools.pdf_reader import read_pdf, PDF_READER_TOOL

TOOLS = [WEB_SEARCH_TOOL, FETCH_PAGE_TOOL, ARXIV_SEARCH_TOOL, PDF_READER_TOOL]
TOOL_FUNCTIONS = {
    "web_search": web_search,
    "fetch_page": fetch_page,
    "arxiv_search": arxiv_search,
    "read_pdf": read_pdf,
}

MAX_EXECUTOR_ITERATIONS = 6
MAX_CRITIC_RETRIES = 2


class ResearchOrchestrator:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()

    def run(self, query: str) -> Generator[dict, None, None]:
        """
        Run the full research pipeline. Yields status dicts so the UI can stream progress.
        Each dict has: {"type": "status"|"note"|"report"|"error", "content": str}
        """
        self.short_term.clear()
        self.short_term.set_query(query)

        # --- Step 1: Retrieve relevant past research ---
        yield {"type": "status", "content": "Checking long-term memory for relevant past research..."}
        past = self.long_term.retrieve(query, n_results=3)
        past_block = ""
        if past:
            past_block = "\n\n".join(
                f"[Past research on '{r['original_query']}']\n{r['content']}" for r in past
            )
            yield {"type": "status", "content": f"Found {len(past)} relevant past notes."}
        else:
            yield {"type": "status", "content": "No past research found. Starting fresh."}

        # --- Step 2: Plan ---
        yield {"type": "status", "content": "Planning research sub-tasks..."}
        sub_tasks = self._plan(query)
        self.short_term.set_sub_tasks(sub_tasks)
        yield {"type": "status", "content": f"Plan: {len(sub_tasks)} sub-tasks identified."}
        for t in sub_tasks:
            yield {"type": "status", "content": f"  • {t}"}

        # --- Step 3: Execute (with critic retry loop) ---
        for attempt in range(MAX_CRITIC_RETRIES + 1):
            yield {"type": "status", "content": f"Executing research (attempt {attempt + 1})..."}
            for event in self._execute(query, sub_tasks, past_block):
                yield event

            # --- Step 4: Critic ---
            yield {"type": "status", "content": "Evaluating research completeness..."}
            evaluation = self._critique(query, sub_tasks)
            yield {"type": "status", "content": f"Critic: {evaluation.get('reason', '')}"}

            if evaluation.get("complete"):
                break

            if attempt < MAX_CRITIC_RETRIES:
                missing = evaluation.get("missing", [])
                yield {"type": "status", "content": f"Research incomplete. Filling gaps: {missing}"}
                sub_tasks = missing  # retry with only the missing tasks
                self.short_term.set_sub_tasks(sub_tasks)
                self.short_term.completed_tasks = []

        # --- Step 5: Synthesize ---
        yield {"type": "status", "content": "Synthesizing report..."}
        report = self._synthesize(query)

        # --- Step 6: Persist to long-term memory ---
        self.long_term.save_session_summary(query, report)
        for note in self.short_term.notes:
            self.long_term.save(query, note.content, note.source, note.tool_used)

        yield {"type": "report", "content": report}

    def _plan(self, query: str) -> list[str]:
        resp = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=PLANNER_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        text = resp.content[0].text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        try:
            tasks = json.loads(text)
            return tasks if isinstance(tasks, list) else [query]
        except json.JSONDecodeError:
            return [query]

    def _execute(self, query: str, sub_tasks: list[str], past_block: str) -> Generator[dict, None, None]:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Research query: {query}\n\n"
                    f"Work through these sub-tasks one by one using your tools:\n"
                    + "\n".join(f"- {t}" for t in sub_tasks)
                ),
            }
        ]

        for _ in range(MAX_EXECUTOR_ITERATIONS):
            pending = self.short_term.pending_tasks()
            if not pending:
                break

            system = EXECUTOR_SYSTEM_PROMPT.format(
                query=query,
                sub_tasks="\n".join(f"- {t}" for t in sub_tasks),
                past_notes=past_block or "None",
                current_notes=self.short_term.get_context_block(),
            )

            resp = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=system,
                tools=TOOLS,
                messages=messages,
            )

            # Process tool calls
            tool_called = False
            for block in resp.content:
                if block.type == "tool_use":
                    tool_called = True
                    tool_name = block.name
                    tool_input = block.input
                    yield {"type": "status", "content": f"Using tool: {tool_name}({list(tool_input.keys())})"}

                    fn = TOOL_FUNCTIONS.get(tool_name)
                    if fn:
                        try:
                            result = fn(**tool_input)
                        except Exception as e:
                            result = {"error": str(e)}
                    else:
                        result = {"error": f"Unknown tool: {tool_name}"}

                    result_str = json.dumps(result, ensure_ascii=False)[:4000]

                    # Store as short-term note
                    source = (
                        tool_input.get("url")
                        or tool_input.get("query")
                        or tool_input.get("source")
                        or tool_name
                    )
                    self.short_term.add_note(source=source, content=result_str, tool_used=tool_name)
                    yield {"type": "note", "content": f"[{tool_name}] {source[:80]}"}

                    # Append tool result to messages for next iteration
                    messages.append({"role": "assistant", "content": resp.content})
                    messages.append({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": block.id, "content": result_str}],
                    })
                    break  # process one tool at a time

            if not tool_called:
                # Model gave a text response — mark all pending tasks as done
                text_response = next((b.text for b in resp.content if hasattr(b, "text")), "")
                for task in pending:
                    self.short_term.mark_task_done(task)
                if text_response:
                    self.short_term.add_note(source="agent_reasoning", content=text_response, tool_used="reasoning")
                break

    def _critique(self, query: str, sub_tasks: list[str]) -> dict:
        prompt = CRITIC_PROMPT.format(
            query=query,
            sub_tasks="\n".join(f"- {t}" for t in sub_tasks),
            completed_tasks="\n".join(f"- {t}" for t in self.short_term.completed_tasks),
            notes=self.short_term.get_context_block()[:3000],
        )
        resp = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"complete": True, "reason": "Could not parse critic response — proceeding."}

    def _synthesize(self, query: str) -> str:
        prompt = SYNTHESIZER_PROMPT.format(
            query=query,
            notes=self.short_term.get_context_block(),
        )
        resp = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
