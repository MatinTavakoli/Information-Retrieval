import os
import sys
import time
import math
import numpy as np
from heapq import heappop, heappush, heapify

inverted_index = {}
champion_list = {}
doc_lengths = {}
stop_words = [
    "از",
    "این",
    "آن",
    "به",
    "با",
    "بر",
    "برای",
    "پس",
    "تا",
    "در",
    "را",
    "که",
    "و",
]

punctuations = [
    ".",
    "،",
    "؛",
    ":",
    "؟",
    "!",
    "»",
    "«",
]

connected_pronouns = [
    "ام",
    "ات",
    "اش",
    "مان",
    "تان",
    "شان",
]


# phase 1 functions


# a function for removing the last occurrence of a pattern
# returns the new string, with the last occurrence removed
def replace_last(string, target, sub):
    reverse_target = target[::-1]
    reverse_sub = sub[::-1]

    return string[::-1].replace(reverse_target, reverse_sub, 1)[::-1]


# tokenize all words in current document
# returns a set, which will be merged with previously founded tokens
def tokenize(plain_text):
    return set([token for token in plain_text.split()])


# makes a dictionary, where for every token, an array of doc_ids is created
# phase 2: term_frequencies dictionary filled
# returns this dictionary
def build_inverted_index(tokens, docs):
    global inverted_index
    global doc_lengths

    for token in tokens:
        inverted_index[token] = {}
        for doc_id in docs:
            if token in docs[doc_id].split():
                tf = compute_tf(token, docs[doc_id])
                inverted_index[token][doc_id] = tf
    for doc_id in docs:
        doc_lengths[doc_id] = compute_doc_length(docs, doc_id)


# stage 1 of normalization ("حذف "ها)
def normalization_1():
    global inverted_index

    for token in list(inverted_index.keys()):
        if "ها" in token and "\u200c" in token:
            token_1 = token.replace("\u200c", "")
            if "های" in token:
                reduced_token = replace_last(token_1, "های", "")
            else:
                reduced_token = replace_last(token_1, "ها", "")
            if reduced_token in inverted_index.keys():
                inverted_index[reduced_token] = list(set().union(inverted_index[reduced_token], inverted_index[token]))
                inverted_index.pop(token)
            else:
                inverted_index[reduced_token] = inverted_index[token]
                inverted_index.pop(token)


# stage 2 of normalization ("حذف "تر و ترین)
def normalization_2():
    global inverted_index

    for token in list(inverted_index.keys()):
        if "تر" in token and "\u200c" in token:
            token_1 = token.replace("\u200c", "")
            if "ترین" in token:
                reduced_token = replace_last(token_1, "ترین", "")
            else:
                reduced_token = replace_last(token_1, "تر", "")
            if reduced_token in inverted_index.keys():
                inverted_index[reduced_token] = list(set().union(inverted_index[reduced_token], inverted_index[token]))
                inverted_index.pop(token)
            else:
                inverted_index[reduced_token] = inverted_index[token]
                inverted_index.pop(token)


# stage 3 of normalization ("حذف "بی)
def normalization_3():
    global inverted_index

    for token in list(inverted_index.keys()):
        if "بی" in token and "\u200c" in token:
            token_1 = token.replace("\u200c", "")
            reduced_token = token_1.replace("بی", "", 1)
            if reduced_token in inverted_index.keys():
                inverted_index[reduced_token] = list(set().union(inverted_index[reduced_token], inverted_index[token]))
                inverted_index.pop(token)
            else:
                inverted_index[reduced_token] = inverted_index[token]
                inverted_index.pop(token)


# stage 4 of normalization (حذف ضمیر متصل)
def normalization_4():
    global inverted_index
    global connected_pronouns

    for token in list(inverted_index.keys()):
        for cp in connected_pronouns:
            if cp in token and "\u200c" in token:
                token_1 = token.replace("\u200c", "")
                reduced_token = replace_last(token_1, cp, "")
                if reduced_token in inverted_index.keys():
                    inverted_index[reduced_token] = list(
                        set().union(inverted_index[reduced_token], inverted_index[token]))
                    inverted_index.pop(token)
                else:
                    inverted_index[reduced_token] = inverted_index[token]
                    inverted_index.pop(token)
                break


# stage 5 of normalization (حذف علائم نگارشی)
def normalization_5():
    global inverted_index
    global punctuations

    for token in list(inverted_index.keys()):
        if '.' in token:  # if it is at the end of a line
            reduced_token = token[:-1]  # cloning token
            for p in punctuations:
                if p in token:
                    reduced_token = reduced_token.replace(p, "")
            if reduced_token in inverted_index.keys() and inverted_index[reduced_token] is not None:
                inverted_index[reduced_token] = list(set().union(inverted_index[reduced_token], inverted_index[token]))
                inverted_index.pop(token)
            else:
                inverted_index[reduced_token] = inverted_index[token]
                inverted_index.pop(token)


# removes terms that add no meaning (i.e. "و", "با", "برای", etc.)
def remove_stop_words():
    global inverted_index
    global stop_words

    for sw in stop_words:
        inverted_index.pop(sw, "None")


# returns the list of documents which contains the single word query
# returns an error message if no occurrences were found
def single_word_query_doc_retriever(query):
    global inverted_index
    if query not in inverted_index.keys():
        return "no occurrences were found"
    return inverted_index[query]


# returns the list of documents which contains the multi word query, sorted in descending order
def multi_word_query_doc_retriever(query, num_of_docs):
    global inverted_index
    doc_scores = []
    for i in range(num_of_docs):
        doc_scores.append([i, 0])
    for word in query.split():
        if word in inverted_index.keys():
            for doc_id in inverted_index[word]:
                doc_scores[doc_id][1] = doc_scores[doc_id][1] + 1

    # sorting docs by their scores (bubble sort. TODO: use a better sorting algorithm!)
    for i in range(num_of_docs):
        for j in range(num_of_docs - 1, i - 1, -1):
            if doc_scores[j][1] > doc_scores[j - 1][1]:
                tmp = doc_scores[j]
                doc_scores[j] = doc_scores[j - 1]
                doc_scores[j - 1] = tmp

    return doc_scores


# phase 2 functions

# makes a dictionary, where for every token, the top k suitable doc_ids are stored
# returns this dictionary
def build_champion_lists(tokens, docs):
    global inverted_index
    global doc_lengths
    global champion_list

    for token in tokens:
        champion_list[token] = {}
        for doc_id in inverted_index[token].keys():
            champion_list[token][doc_id] = inverted_index[token][doc_id] / doc_lengths[doc_id]
        champion_list[token] = sorted(champion_list[token].items(), key=lambda kv: kv[1], reverse=True)  # sorting


def compute_doc_length(docs, doc_id):
    global inverted_index
    len = 0
    for word in docs[doc_id].split():
        len += inverted_index[word][doc_id] ** 2
    return len ** (1 / 2)


def compute_tf(term, doc):
    term_frequency = 0
    for t in doc.split():
        if term == t:
            term_frequency = term_frequency + 1
    # return term_frequency
    return 1 + math.log10(term_frequency)


def compute_idf(term, docs):
    global inverted_index
    # return len(inverted_index[term])
    return math.log10(len(docs) / len(inverted_index[term]))


def compute_tf_idf(term, doc, docs):
    return compute_tf(term, doc) * compute_idf(term, docs)


def compute_cosine_similarity(query, docs, doc_id):
    global doc_lengths
    res = 0
    for word in query.split():
        if word in docs[doc_id].split():
            res += compute_tf_idf(word, docs[doc_id], docs) * inverted_index[word][doc_id]
    res /= doc_lengths[doc_id]
    return res


def compute_scores(query, docs):
    global champion_list
    scores = {}
    for word in query.split():
        if word in champion_list.keys():
            for doc_tuple in champion_list[word]:
                doc_id = doc_tuple[0]
                if doc_id not in scores.keys():
                    scores[doc_id] = 0
                scores[doc_id] += compute_cosine_similarity(word, docs, doc_id)
        # scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)  # sorting
    return scores


def select_best_k_scores(query, docs, k=3):
    scores = compute_scores(query, docs)
    print('original scores are: {}'.format(scores))
    best_scores = {}
    heap = []
    for doc_id in scores.keys():
        heappush(heap, (-scores[doc_id], doc_id))
    for i in range(k):
        max, best_doc_id = heappop(heap)
        best_scores[best_doc_id] = -max
    return best_scores


def main():
    print("Welcome to my Search Engine!")
    time.sleep(1)

    docs = {}

    path = '.\docs'
    for dirpath, dirnames, files in os.walk(path):
        for i, file_name in enumerate(files):
            f = open(path + '\\' + file_name, "r", encoding="utf8")
            file_index = int(file_name[:-4])
            docs[file_index] = f.read()  # stores value "doc" in key "i"

    # raw tokens
    tokens = set()
    for doc_id in docs:
        tokens = tokens.union(tokenize(docs[doc_id]))
    # print("-----------------------------------")
    # print("raw tokens: ")
    # print("-----------------------------------")
    # time.sleep(1)
    # print(tokens)

    # time.sleep(0.75)
    # print()
    # print()
    # print()
    # time.sleep(0.75)

    # inverted index
    build_inverted_index(tokens, docs)
    # print(inverted_index["هزار"])
    build_champion_lists(tokens, docs)

    # print("-----------------------------------")
    # print("inverted index: ")
    # print("-----------------------------------")
    # time.sleep(1)
    # print(inverted_index)

    # time.sleep(0.75)
    # print()
    # print()
    # print()
    # time.sleep(0.75)

    normalization_1()
    # print("-----------------------------------")
    # print("inverted index: (after normalization #1)")
    # print("-----------------------------------")
    # time.sleep(1)
    # print(inverted_index)

    # time.sleep(0.75)
    # print()
    # print()
    # print()
    # time.sleep(0.75)

    normalization_2()
    # print("-----------------------------------")
    # print("inverted index: (after normalization #2)")
    # print("-----------------------------------")
    # time.sleep(1)
    # print(inverted_index)

    # time.sleep(0.75)
    # print()
    # print()
    # print()
    # time.sleep(0.75)

    normalization_3()
    # print("-----------------------------------")
    # print("inverted index: (after normalization #3)")
    # print("-----------------------------------")
    # time.sleep(1)
    # print(inverted_index)

    # time.sleep(0.75)
    # print()
    # print()
    # print()
    # time.sleep(0.75)

    normalization_4()
    # print("-----------------------------------")
    # print("inverted index: (after normalization #4)")
    # print("-----------------------------------")
    # time.sleep(1)
    # print(inverted_index)

    # time.sleep(0.75)
    # print()
    # print()
    # print()
    # time.sleep(0.75)

    # normalization_5()
    # print("-----------------------------------")
    # print("inverted index: (after normalization #5)")
    # print("-----------------------------------")
    # time.sleep(1)
    # print(inverted_index)

    # time.sleep(0.75)
    # print()
    # print()
    # print()
    # time.sleep(0.75)

    remove_stop_words()
    # print("-----------------------------------")
    # print("inverted index: (after eliminating stop words)")
    # print("-----------------------------------")
    # time.sleep(1)
    # print(inverted_index)

    # time.sleep(0.75)
    # print()
    # print()
    # print()
    # time.sleep(0.75)

    # query = "هزار"
    # query = "کاربر"
    query = "سلامت کرونا"
    # query = "لیگ قهرمانان آسیا"
    # query = "مطالب طنز"
    # query = "لیگ اروپا"
    # query = "فجر بهمن سینما"
    # query = "استغفار"
    print('query is {}'.format(query))
    # print(champion_list["هزار"])
    print('best (sorted) scores are: {}'.format(select_best_k_scores(query, docs, k=min(5,
                                                                                        len(compute_scores(query,
                                                                                                           docs).keys())))))


if __name__ == "__main__":
    main()
