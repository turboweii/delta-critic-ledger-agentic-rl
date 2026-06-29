from __future__ import annotations


DATE_CONTEXT = (
    "The current date is 2024-05-15 (Wednesday). "
    "When users mention dates without specifying the year, assume 2024."
)


def tau_system_prompt(wiki):
    return DATE_CONTEXT
