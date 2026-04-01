"""Main agent orchestration loop — implements the Planner → Executor → Critic → Synthesizer pattern."""

import json
import os
from typing import Generator
import google.generativeai as genai

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

MODEL = "gemini-2.0-flash"

TOOL_FUNCTIONS = {
    "web_search": web_search,
    "fetch_page": fetch_page,
    "arxiv_search": arxiv_search,
    "read_pdf": read_pdf,
}

# Convert tool schemas from Anthropic format → Gemini function declaration format
def _to_gemini_tools(schemas: list[dict]) -> list[genai.protos.Tool]:
    declarations = []
    for s in schemas:
        declarations.append(
            genai.protos.FunctionDeclaration(
                name=s["name"],
                description=s["description"],
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        k: genai.protos.Schema(
                            type=genai.protos.Type[v.get("type", "string").upper()],
                            description=v.get("description", ""),
                        )
                        for k, v in s["input_schema"]["properties"].items()
                    },
                    required=s["input_schema"].get("required", []),
                ),
            )
        )
    return [genai.protos.Tool(function_declarations=declarations)]


MAX_EXECUTOR_ITERATIONS = 6
MAX_CRITIC_RETRIES = 2


class ResearchOrchestrator:
    def __init__(self):
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
        self._gemini_tools = _to_gemini_tools(
            [WEB_SEARCH_TOOL, FETCH_PAGE_TOOL, ARXIV_SEARCH_TOOL, PDF_READER_TOOL]
        )

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
                sub_tasks = missing
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
        model = genai.GenerativeModel(MODEL)
        resp = model.generate_content(
            f"{PLANNER_PROMPT}\n\nResearch query: {query}"
        )
        text = resp.text.strip()
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
        model = genai.GenerativeModel(MODEL, tools=self._gemini_tools)
        chat = model.start_chat()

        system = EXECUTOR_SYSTEM_PROMPT.format(
            query=query,
            sub_tasks="\n".join(f"- {t}" for t in sub_tasks),
            past_notes=past_block or "None",
            current_notes="",
        )

        initial_message = (
            f"{system}\n\n"
            f"Begin researching. Work through each sub-task using your tools."
        )

        for iteration in range(MAX_EXECUTOR_ITERATIONS):
            pending = self.short_term.pending_tasks()
            if not pending:
                break

            # Update context with current notes for subsequent iterations
            if iteration == 0:
                msg = initial_message
            else:
                msg = (
                    f"Continue researching. Pending tasks: {pending}\n\n"
                    f"Notes gathered so far:\n{self.short_term.get_context_block()[:2000]}"
                )

            resp = chat.send_message(msg)

            tool_called = False
            for part in resp.parts:
                if part.function_call:
                    tool_called = True
                    fn_name = part.function_call.name
                    fn_args = dict(part.function_call.args)

                    yield {"type": "status", "content": f"Using tool: {fn_name}({list(fn_args.keys())})"}

                    fn = TOOL_FUNCTIONS.get(fn_name)
                    if fn:
                        try:
                            result = fn(**fn_args)
                        except Exception as e:
                            result = {"error": str(e)}
                    else:
                        result = {"error": f"Unknown tool: {fn_name}"}

                    result_str = json.dumps(result, ensure_ascii=False)[:4000]

                    source = (
                        fn_args.get("url")
                        or fn_args.get("query")
                        or fn_args.get("source")
                        or fn_name
                    )
                    self.short_term.add_note(source=source, content=result_str, tool_used=fn_name)
                    yield {"type": "note", "content": f"[{fn_name}] {str(source)[:80]}"}

                    # Send tool result back to model
                    chat.send_message(
                        genai.protos.Content(
                            parts=[
                                genai.protos.Part(
                                    function_response=genai.protos.FunctionResponse(
                                        name=fn_name,
                                        response={"result": result_str},
                                    )
                                )
                            ]
                        )
                    )
                    break  # one tool at a time

            if not tool_called:
                # Model gave a text response — mark pending tasks done
                for task in pending:
                    self.short_term.mark_task_done(task)
                text_resp = resp.text if hasattr(resp, "text") else ""
                if text_resp:
                    self.short_term.add_note(
                        source="agent_reasoning", content=text_resp, tool_used="reasoning"
                    )
                break

    def _critique(self, query: str, sub_tasks: list[str]) -> dict:
        model = genai.GenerativeModel(MODEL)
        prompt = CRITIC_PROMPT.format(
            query=query,
            sub_tasks="\n".join(f"- {t}" for t in sub_tasks),
            completed_tasks="\n".join(f"- {t}" for t in self.short_term.completed_tasks),
            notes=self.short_term.get_context_block()[:3000],
        )
        resp = model.generate_content(prompt)
        text = resp.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"complete": True, "reason": "Could not parse critic response — proceeding."}

    def _synthesize(self, query: str) -> str:
        model = genai.GenerativeModel(MODEL)
        prompt = SYNTHESIZER_PROMPT.format(
            query=query,
            notes=self.short_term.get_context_block(),
        )
        resp = model.generate_content(prompt)
        return resp.text.strip()
