"""
smartcontent_api.py
~~~~~~~~~~~~~~~~~~~~

This module defines a FastAPI application that exposes a single endpoint for
summarising web pages.  Clients send a JSON payload containing a URL and the
number of sentences they would like returned in the summary.  The service
fetches the page, extracts the visible text, generates a frequency‑based
summary, and returns meta data along with the summary.  This API is designed
for deployment behind the scenes on a cloud platform and can be monetised
through per‑request billing on marketplaces like RapidAPI.

The summarisation algorithm implemented here avoids external model
dependencies (which cannot be downloaded in this environment) and instead
relies on a simple heuristic: sentences are scored by the sum of their word
frequencies across the entire document, and the top‑scoring sentences are
returned.  While rudimentary, this approach produces concise summaries
without requiring additional data downloads.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import List

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="SmartContent API",
              description="Fetches a web page, extracts human‑readable text, "
                          "and returns a simple frequency‑based summary along "
                          "with the page title, meta description and H1 tags.",
              version="1.0.0")


class UrlRequest(BaseModel):
    """Request schema for the summarise endpoint."""

    url: str = Field(..., description="The full URL of the page to summarise")
    num_sentences: int = Field(5, ge=1, le=20,
                               description="Number of sentences to include in the summary")


def _tokenise_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation as delimiters.

    Parameters
    ----------
    text : str
        Raw text extracted from the HTML page.

    Returns
    -------
    List[str]
        A list of sentences.  Empty strings are filtered out.
    """
    # Use regex to split on sentence endings while preserving abbreviations.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _word_frequencies(sentences: List[str]) -> defaultdict[str, int]:
    """Compute a frequency table for all words in the given sentences.

    Stop words are not removed to avoid introducing a heavy dependency on
    language models; the simple frequency count tends to favour sentences with
    more common words, which often correspond to key ideas in the article.

    Parameters
    ----------
    sentences : List[str]
        List of sentences to analyse.

    Returns
    -------
    defaultdict[str, int]
        Mapping of words to their frequency across all sentences.
    """
    freq: defaultdict[str, int] = defaultdict(int)
    for sentence in sentences:
        for word in re.findall(r'\b\w+\b', sentence.lower()):
            freq[word] += 1
    return freq


def _score_sentences(sentences: List[str], freq: defaultdict[str, int]) -> List[tuple[int, str]]:
    """Assign a score to each sentence based on word frequencies.

    Parameters
    ----------
    sentences : List[str]
        Sentences to score.
    freq : defaultdict[str, int]
        Word frequency table.

    Returns
    -------
    List[tuple[int, str]]
        A list of (score, sentence) tuples.
    """
    scored: List[tuple[int, str]] = []
    for sentence in sentences:
        words = re.findall(r'\b\w+\b', sentence.lower())
        score = sum(freq[word] for word in words)
        scored.append((score, sentence))
    return scored


def summarise_text(text: str, num_sentences: int) -> str:
    """Generate a simple summary from raw text.

    This function splits the input into sentences, calculates a word
    frequency table, scores each sentence, and then returns the top
    `num_sentences` sentences concatenated together in original order.

    Parameters
    ----------
    text : str
        The text to summarise.
    num_sentences : int
        The number of sentences to include in the summary.

    Returns
    -------
    str
        Concise summary comprising the highest‑scoring sentences.
    """
    sentences = _tokenise_sentences(text)
    if not sentences:
        return ""
    freq = _word_frequencies(sentences)
    scored = _score_sentences(sentences, freq)
    # Select the top N sentences by score
    top_sentences = sorted(scored, key=lambda pair: pair[0], reverse=True)[:num_sentences]
    # Preserve original order of sentences in the summary
    ranked_sentences = {sent: score for score, sent in top_sentences}
    ordered = [s for s in sentences if s in ranked_sentences]
    return ' '.join(ordered)


@app.post("/summarise")
def summarise(request: UrlRequest):
    """Fetch a page and return a summarised version of its content.

    The endpoint expects a JSON body with a `url` field and an optional
    `num_sentences` field specifying the desired length of the summary.
    It returns the page title, meta description, top‑level headings and
    summary.

    Raises
    ------
    HTTPException
        If the URL cannot be fetched or the response status code is not 200.
    """
    try:
        resp = requests.get(request.url, timeout=15)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error fetching URL: {exc}")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code,
                            detail=f"Non‑200 status code returned: {resp.status_code}")
    soup = BeautifulSoup(resp.text, "html.parser")
    # Remove non‑visible elements
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    # Extract text and clean it up
    text = soup.get_text(separator=' ')
    text = re.sub(r'\s+', ' ', text).strip()
    summary = summarise_text(text, request.num_sentences)
    # Extract metadata
    title = soup.title.get_text(strip=True) if soup.title else ''
    description = ''
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc and meta_desc.get('content'):
        description = meta_desc['content'].strip()
    headings = [tag.get_text(strip=True) for tag in soup.find_all('h1')]
    return {
        "title": title,
        "description": description,
        "headings": headings,
        "summary": summary
    }
