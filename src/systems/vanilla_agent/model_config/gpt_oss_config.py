"""
GPT-OSS model configuration with OpenAI Harmony format.

This module provides prompts and message building functions for GPT-OSS models
that follow the OpenAI Harmony format, where instructions go in "developer" role
and system message contains meta information.

Reference: https://cookbook.openai.com/articles/openai-harmony
"""

import re
from datetime import datetime, timezone
from typing import List
from openai.types.chat import ChatCompletionMessageParam
from tools.web_search import SearchResult


def is_gpt_oss_model(model_id: str) -> bool:
    """Check if the model_id indicates a gpt-oss model.

    Args:
        model_id: The model identifier string

    Returns:
        True if model_id matches gpt.*oss pattern
    """
    if not model_id:
        return False
    # Match pattern like 'gpt-4o-mini-oss', 'gpt-4-oss', 'gpt-3.5-turbo-oss', etc.
    return bool(re.search(r'gpt.*oss', model_id, re.IGNORECASE))


def SYSTEM_MESSAGE(enable_think: bool = True) -> str:
    """System message for GPT-OSS containing meta information.

    In Harmony format, system message contains reasoning effort, dates, and channels.

    Args:
        enable_think: Whether to enable thinking mode for the model

    Returns:
        The formatted system message
    """
    reasoning_level = "high" if enable_think else "low"
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return f"""You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-12
Current date: {current_date}

Reasoning: {reasoning_level}

# Valid channels: analysis, commentary, final. Channel must be included for every message."""


def ANSWER_PROMPT() -> str:
    """Developer message for answering questions based on search results.

    Returns:
        The formatted developer prompt
    """
    return """# Instructions

You are a knowledgeable AI search assistant built by the RMIT IR team.

Your search engine has returned a list of relevant webpages based on the user's question, listed below in <search-results> tags. These webpages are your knowledge.

The next user message is the full user question, and you need to explain and answer the question based on the search results. Do not make up answers that are not supported by the search results. If the search results do not have the necessary information for you to answer the search question, say you don't have enough information for the question.

Try to provide a balanced view for controversial topics.

Tailor the complexity of your response to the user question, use simpler bullet points for simple questions, and sections for more detailed explanations for complex topics or rich content.

Do not answer to greetings or chat with the user, always reply in English.


Format guidelines:

1. Cite information from the search results using the following format: `【{ID}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.
2. Use back ticks to format any code snippets.
3. Use latex format to write any mathematical expressions, put inline math between `\\(...\\)` and display math between `\\[...\\]`.

"""


def REVIEW_DOCUMENTS_PROMPT(question: str, next_query: str, current_time: datetime, query_history_section: str = "") -> str:
    """Developer message for reviewing document relevance and sufficiency.

    Args:
        question: The original user question
        next_query: The current search query being evaluated
        current_time: Current datetime for context
        query_history_section: Formatted query history string

    Returns:
        The formatted review prompt for GPT-OSS
    """
    return f"""# Instructions

You are an expert in answering user question "{question}". We are doing research on user's question and currently working on aspect "{next_query}"

Current time at UTC+00:00 timezone: {current_time}

Go through each document in the search results and evaluate their relevance.

1. Does the user want a simple answer or a comprehensive explanation? For comprehensive explanations, we may need searching with different aspects to cover a wider range of perspectives.
2. For controversial, convergent, divergent, evaluative, complex systemic, ethical-moral, problem-solving, or recommendation questions, consider whether multiple aspects would form a more balanced, comprehensive view.
3. Identify unique, new documents that are important for answering the question but not included in previous documents. List their IDs (# in ID=[#]) in a comma-separated format within <useful-docs> xml tags. If multiple documents are similar, choose the one with better quality. Do not provide duplicated documents from previous turns. If no new documents are useful, leave <useful-docs></useful-docs> empty.
4. Evaluate whether all collected documents (including previous turns) fully address the user's query and any sub-components. Consider what information is still missing or uncertain.
5. Determine <is-sufficient>: When you answer 'yes', we will proceed to generate the final answer. If you answer 'no', we will continue with a new search query.

Response Logic:

1. If <useful-docs> is not empty, provide a brief summary within <useful-docs-summary> xml tags (1-2 sentences). Start with "These documents discuss..." and indicate whether they are sufficient or still missing information. Do not mention specific document IDs in the summary.
2. If information is missing or uncertain, return 'no' in <is-sufficient> and generate a new query in <new-query> xml tags indicating what clarification is needed.
3. If current search results have very limited information, use different techniques: query expansion, query relaxation, query segmentation, synonyms, or reasonable guesses with different keywords. Put the new query in <new-query> xml tags. Do not repeat previous unsuccessful queries.
4. Your purpose is to judge document relevance, not to provide the final answer yet.

Response Format:

- <useful-docs>1,2,3</useful-docs> (list of document IDs that are useful for answering the question, separated by commas)
- <useful-docs-summary></useful-docs-summary> (short summary of what these useful documents are talking about and what is missing, just the summary, only if useful-docs is not empty)
- <is-sufficient>yes or no</is-sufficient> (For all of the documents we have collected, including previous documents, do we have enough information to answer the user question?)
- <new-query>your new query</new-query> (if is-sufficient is 'no')

{query_history_section}

Here is the current search query: "{next_query}"
Here is the search results for current search query:

"""


def QUERY_VARIANTS_PROMPT(num_qvs: int, enable_think: bool = True) -> str:
    """Developer message for generating query variants.

    Args:
        num_qvs: Number of query variants to generate
        enable_think: Whether to enable thinking mode

    Returns:
        The formatted query variants prompt for GPT-OSS
    """
    return f"""# Instructions

You will receive a question from a user and you need interpret what the question is actually asking about and come up with 2 to {num_qvs} Google search queries to answer that question.

Try express the same question in different ways, use different techniques, query expansion, query relaxation, query segmentation, use different synonyms, use reasonable guess and different keywords to reach different aspects.

Try to provide a balanced view for controversial topics.

Only provide English queries, no matter what language the user question is in.

Current time at UTC+00:00 timezone: {datetime.now(timezone.utc)}

To comply with the format, put your query variants enclosed in queries xml markup:

<queries>
query variant 1
query variant 2
...
</queries>

Put each query in a line, do not add any prefix on each query, only provide the query themselves."""


def REFORMULATE_QUERY_PROMPT() -> str:
    """Developer message for reformulating a query.

    Returns:
        The formatted reformulation prompt for GPT-OSS
    """
    return """# Instructions

You will receive a question from a user and you need interpret what the question is actually asking about and come up with a better Google search query to answer that question. Only provide the reformulated query, do not add any prefix or suffix."""


def build_to_context(results: List[SearchResult]) -> str:
    """Build context string from search results.

    Args:
        results: List of SearchResult objects

    Returns:
        Formatted context string with search results
    """
    context = "<search-results>"
    context += "\n".join([f"""
Webpage ID=[{r.sid}] URL=[{r.url}] Date=[{r.date}]:

{r.text}""" for r in results])
    context += "</search-results>"
    return context


def build_answer_messages(
    results: str | List[SearchResult],
    query: str,
    enable_think: bool = True
) -> List[ChatCompletionMessageParam]:
    #! Assuming using Cerebras, system role is mapped to developer role.
    """Build LLM messages in OpenAI Harmony format for GPT-OSS.

    In OpenAI Harmony format:
    - system message contains meta information (reasoning level, dates, channels)
    - developer message contains the actual instructions and context
    - user message contains the query

    Args:
        results: Either a string context or list of SearchResult objects
        query: The user's question
        enable_think: Whether to enable thinking mode

    Returns:
        List of chat completion messages in Harmony format
    """
    if isinstance(results, list):
        if len(results) > 0 and isinstance(results[0], SearchResult):
            context = build_to_context(results)
        else:
            context = "<search-results>Empty results</search-results>"
    elif isinstance(results, str):
        context = results

    developer_content = ANSWER_PROMPT() + context

    return [
        {"role": "system", "content": developer_content},
        {"role": "user", "content": query},
    ]
