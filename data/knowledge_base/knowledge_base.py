import re
from nltk.tokenize import sent_tokenize
from transformers import GPT2TokenizerFast
from typing import Set
import numpy as np
# import openai
import pandas as pd
import wikipedia
import pickle
import tiktoken
import os

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

# openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = """Answer the question as truthfully as possible, and if you're unsure of the answer, say "Sorry, I don't know".

Q: 
A:"""

# def ask(prompt):
#     answer = openai.Completion.create(
#         prompt=prompt,
#         temperature=0,
#         max_tokens=300,
#         model=COMPLETIONS_MODEL
#     )
#     answer = answer["choices"][0]["text"].strip(" \n")

#     return answer


def filterWikiPages(titles):
    """
    Get the titles which are related to keywords, given a list of titles
    """
    titles = [title for title in titles if 'marijuana' in title.lower(
    ) or 'weed' in title.lower() or 'cannabis' in title.lower()]
    # t = []
    # for title in titles:
    #     words = ["weed", "marijuana", "cannabis", "legalization"]
    #     for word in words:
    #         if word in title.lower():
    #             t.append(title)

    return titles


def getWikiPage(title):
    """
    Get the wikipedia page given a title
    """
    try:
        return wikipedia.page(title)
    except wikipedia.exceptions.DisambiguationError as e:
        return None
    except wikipedia.exceptions.PageError as e:
        return None


num = 0


def recursivelyFindAllPages(titles, titles_so_far=set()):
    """
    Recursively find all the pages that are linked to the Wikipedia titles in the list
    """
    all_pages = []
    global num
    titles = list(set(titles) - titles_so_far)
    titles = filterWikiPages(titles)
    # print("Titles #" + str(num) + ": " + titles[num])
    titles_so_far.update(titles)
    for title in titles:
        page = getWikiPage(title)
        if page is None:
            continue
        all_pages.append(page)
        print("All_page len #" + str(num) + ": " + str(len(all_pages)))
        try:
            new_pages = recursivelyFindAllPages(page.links, titles_so_far)
        except:
            # print("Return: " + str(len(all_pages)) + " ex - " + all_pages[12])
            print("terminating #" + str(num) +
                  " length: " + str(len(all_pages)))
            return all_pages
        for pg in new_pages:
            if pg.title not in [p.title for p in all_pages]:
                all_pages.append(pg)
        titles_so_far.update(page.links)
    if num > 50:
        print("Return: " + str(len(all_pages)) + " ex - " + all_pages[12])
        return all_pages
    num += 1
    print("Return: " + str(len(all_pages)))
    return all_pages

# for page in pages:
    # print(page.title)
    # print(page.)
# print(len(pages))

# filtering wiki pages


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def count_tokens(text):
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))


def reduce_long(
    long_text: str, long_text_tokens: bool = False, max_len: int = 590
) -> str:
    """
    Reduce a long text to a maximum of `max_len` tokens by potentially cutting at a sentence end
    """
    if not long_text_tokens:
        long_text_tokens = count_tokens(long_text)
    if long_text_tokens > max_len:
        sentences = sent_tokenize(long_text.replace("\n", " "))
        ntokens = 0
        for i, sentence in enumerate(sentences):
            ntokens += 1 + count_tokens(sentence)
            if ntokens > max_len:
                return ". ".join(sentences[:i][:-1]) + "."

    return long_text


discard_categories = ['See also', 'References', 'External links', 'Further reading', "Footnotes",
                      "Bibliography", "Sources", "Citations", "Literature", "Footnotes", "Notes and references",
                      "Photo gallery", "Works cited", "Photos", "Gallery", "Notes", "References and sources",
                      "References and notes", ]


def extract_sections(
    wiki_text: str,
    title: str,
    max_len: int = 1500,
    discard_categories: Set[str] = discard_categories,
) -> str:
    """
    Extract the sections of a Wikipedia page, discarding the references and other low information sections
    """
    if len(wiki_text) == 0:
        return []

    # find all headings and the coresponding contents
    headings = re.findall("==+ .* ==+", wiki_text)
    for heading in headings:
        wiki_text = wiki_text.replace(heading, "==+ !! ==+")
    contents = wiki_text.split("==+ !! ==+")
    contents = [c.strip() for c in contents]
    assert len(headings) == len(contents) - 1

    cont = contents.pop(0).strip()
    outputs = [(title, "Summary", cont, count_tokens(cont)+4)]

    # discard the discard categories, accounting for a tree structure
    max_level = 100
    keep_group_level = max_level
    remove_group_level = max_level
    nheadings, ncontents = [], []
    for heading, content in zip(headings, contents):
        plain_heading = " ".join(heading.split(" ")[1:-1])
        num_equals = len(heading.split(" ")[0])
        if num_equals <= keep_group_level:
            keep_group_level = max_level

        if num_equals > remove_group_level:
            if (
                num_equals <= keep_group_level
            ):
                continue
        keep_group_level = max_level
        if plain_heading in discard_categories:
            remove_group_level = num_equals
            keep_group_level = max_level
            continue
        nheadings.append(heading.replace("=", "").strip())
        ncontents.append(content)
        remove_group_level = max_level

    # count the tokens of each section
    ncontent_ntokens = [
        count_tokens(c)
        + 3
        + count_tokens(" ".join(h.split(" ")[1:-1]))
        - (1 if len(c) == 0 else 0)
        for h, c in zip(nheadings, ncontents)
    ]

    # Create a tuple of (title, section_name, content, number of tokens)
    outputs += [(title, h, c, t) if t < max_len
                else (title, h, reduce_long(c, max_len), count_tokens(reduce_long(c, max_len)))
                for h, c, t in zip(nheadings, ncontents, ncontent_ntokens)]

    return outputs

# recursivelyFindAllPages("")