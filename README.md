# WellBeingChatBot - Exercise
This project creates a Python-powered backend that supports meaningful wellbeing conversations. It uses user state and curated wellbeing insights to craft compassionate responses and suggest helpful next steps.


# Task

Design a Python backend workflow that powers a **personalised wellbeing conversation**. The service should combine user state and curated wellbeing content to produce an empathetic response plus a short, actionable plan for the upcoming day.

You may choose libraries, tools, and model providers freely as long as the solution is written in Python and uses an LLM orchestration framework that is provider-agnostic and supports open-source models. Aim for a focused, production-minded solution that can be implemented within ~1–4 hours.

---

# Core objectives

* **Orchestrate a multi-step LLM workflow** that turns multi-turn user input into:

  * an empathetic conversational reply, and
  * a concise action plan with explanation for why each action was recommended.
* **Integrate structured wellbeing knowledge.** Options include using the provided `dummy_wellbeing_content.json`, creating a small dataset, mocking an external API, or building a simple retrieval flow. Document your choice in the README.
* **Add reflection signals.** Include at least one component that analyses the conversation (e.g., sentiment or theme extraction) to inform the planner.
* **Expose the workflow through a minimal interface** — pick one:

  * **Option A:** Web API (FastAPI, Flask, etc.) with at least one endpoint, or
  * **Option B:** CLI or Streamlit script that runs the flow end-to-end with sample inputs.
* **Implement structured logging** — e.g., a minimal database to log conversations, sessions, and checkpoints.

---

# Functional requirements

1. Accept user context (JSON payload or CLI args) including at minimum:

   * free-text input
   * optional preferences/constraints (available time, focus area, current mood, etc.)
2. Run the context through your orchestration flow which may include:

   * prompt templates or graph/chain nodes
   * tool calls (retrieval, calculators, custom functions)
   * guardrails/validation logic to ensure suggestions are safe and achievable
3. Return one or both of:

   * an empathetic conversational message, and/or
   * a structured plan object (JSON or dataclass) containing 1–2 recommended actions plus brief explainability
4. Log each step with timestamps and relevant metadata.

---

# Technical requirements

* Target **Python 3.10+**.
* Use an LLM orchestration/back-end framework (examples: LangChain, LangGraph, llama-index, etc.), plus pydantic for data validation where appropriate.
* You may call any LLM (OpenAI, Anthropic, HF checkpoints, ollama). Document model configuration and required API keys.
* Include dependency manifest: `requirements.txt` **or** `pyproject.toml`.
* Provide a reproducible run path (environment variables, setup commands, sample invocation).

---

# Deliverables

Bundle your work as a `.zip` containing:

* Source code for the workflow and chosen minimal interface.
* `README.md` including:

  * architecture overview and reasoning
  * setup instructions and API key configuration
  * how to run the workflow (commands, endpoints, example payloads)
  * testing instructions (if applicable)
* Dependency manifest (`requirements.txt` or `pyproject.toml`).
* Any auxiliary data files (e.g., modified or extended wellbeing dataset).
* Optional: automated tests or notebooks demonstrating the flow.

---

# Evaluation criteria

Submissions will be evaluated on:

* **Architecture & workflow design** — clarity, modularity, and effective use of tooling.
* **LLM prompting & reasoning quality** — prompt engineering, guardrails, and edge-case handling.
* **Integration & data handling** — how structured content or external knowledge is incorporated.
* **Code quality & maintainability** — Python coding standards, error handling, logging, and documentation.
* **Reproducibility** — ease of setup and clarity of run instructions.
* **Product thinking** — alignment with a health-tech wellbeing experience.
* **Originality** — creative ideas and thoughtful trade-offs.

---

# Getting started

* The repository includes a sample wellbeing content file `dummy_wellbeing_content.json`. You may extend or replace it.
* Scaffold the project as you prefer (single script, package layout, etc.). Keep scope aligned to a <4 hour implementation.

---

# Suggested timeline

Aim for **1.5–4 hours** of focused implementation. Thoughtful scope decisions are valued over exhaustive features.

---

# Submission instructions

1. Verify the project runs locally using the commands documented in the README.
2. Zip the repository (exclude large virtualenv folders).
3. Share the archive plus a short note summarizing design choices and known trade-offs.

