import os
import sys
import time
import math
import numpy as np
from heapq import heappop, heappush, heapify
import json

inverted_index = {}
champion_list = {}
doc_lengths = {}
centroids = {}
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
# phase 3: cluster mode: determines whether we want an inverted index for all docs, or docs of one particular subject
# phase 3: subject: determines the subject of the cluster (only in cluster_mode=True)
# returns this dictionary
def build_inverted_index(tokens, docs, cluster_mode=False, subject=None):
    global inverted_index
    global doc_lengths

    if not cluster_mode:
        i_i = inverted_index
        d_l = doc_lengths
    else:
        i_i = inverted_index[subject]
        d_l = doc_lengths[subject]

    for token in tokens:
        i_i[token] = {}
        for doc_id in docs:
            if token in docs[doc_id].split():
                tf = compute_tf(token, docs[doc_id])
                i_i[token][doc_id] = tf
    for doc_id in docs:
        d_l[doc_id] = compute_doc_length(docs, doc_id, cluster_mode, subject)


# stage 1 of normalization ("حذف "ها)
# phase 3: cluster mode: determines whether we want an inverted index for all docs, or docs of one particular subject
# phase 3: subject: determines the subject of the cluster (only in cluster_mode=True)
def normalization_1(cluster_mode=False, subject=None):
    global inverted_index

    if not cluster_mode:
        i_i = inverted_index
    else:
        i_i = inverted_index[subject]

    for token in list(i_i.keys()):
        if "ها" in token and "\u200c" in token:
            token_1 = token.replace("\u200c", "")
            if "های" in token:
                reduced_token = replace_last(token_1, "های", "")
            else:
                reduced_token = replace_last(token_1, "ها", "")
            if reduced_token in i_i.keys():
                i_i[reduced_token] = list(set().union(i_i[reduced_token], i_i[token]))
                i_i.pop(token)
            else:
                i_i[reduced_token] = i_i[token]
                i_i.pop(token)


# stage 2 of normalization ("حذف "تر و ترین)
# phase 3: cluster mode: determines whether we want an inverted index for all docs, or docs of one particular subject
# phase 3: subject: determines the subject of the cluster (only in cluster_mode=True)
def normalization_2(cluster_mode=False, subject=None):
    global inverted_index

    if not cluster_mode:
        i_i = inverted_index
    else:
        i_i = inverted_index[subject]

    for token in list(i_i.keys()):
        if "تر" in token and "\u200c" in token:
            token_1 = token.replace("\u200c", "")
            if "ترین" in token:
                reduced_token = replace_last(token_1, "ترین", "")
            else:
                reduced_token = replace_last(token_1, "تر", "")
            if reduced_token in i_i.keys():
                i_i[reduced_token] = list(set().union(i_i[reduced_token], i_i[token]))
                i_i.pop(token)
            else:
                i_i[reduced_token] = i_i[token]
                i_i.pop(token)


# stage 3 of normalization ("حذف "بی)
# phase 3: cluster mode: determines whether we want an inverted index for all docs, or docs of one particular subject
# phase 3: subject: determines the subject of the cluster (only in cluster_mode=True)
def normalization_3(cluster_mode=False, subject=None):
    global inverted_index

    if not cluster_mode:
        i_i = inverted_index
    else:
        i_i = inverted_index[subject]

    for token in list(i_i.keys()):
        if "بی" in token and "\u200c" in token:
            token_1 = token.replace("\u200c", "")
            reduced_token = token_1.replace("بی", "", 1)
            if reduced_token in i_i.keys():
                i_i[reduced_token] = list(set().union(i_i[reduced_token], i_i[token]))
                i_i.pop(token)
            else:
                i_i[reduced_token] = i_i[token]
                i_i.pop(token)


# stage 4 of normalization (حذف ضمیر متصل)
# phase 3: cluster mode: determines whether we want an inverted index for all docs, or docs of one particular subject
# phase 3: subject: determines the subject of the cluster (only in cluster_mode=True)
def normalization_4(cluster_mode=False, subject=None):
    global inverted_index
    global connected_pronouns

    if not cluster_mode:
        i_i = inverted_index
    else:
        i_i = inverted_index[subject]

    for token in list(i_i.keys()):
        for cp in connected_pronouns:
            if cp in token and "\u200c" in token:
                token_1 = token.replace("\u200c", "")
                reduced_token = replace_last(token_1, cp, "")
                if reduced_token in i_i.keys():
                    i_i[reduced_token] = list(set().union(i_i[reduced_token], i_i[token]))
                    i_i.pop(token)
                else:
                    i_i[reduced_token] = i_i[token]
                    i_i.pop(token)
                break


# stage 5 of normalization (حذف علائم نگارشی)
# phase 3: cluster mode: determines whether we want an inverted index for all docs, or docs of one particular subject
# phase 3: subject: determines the subject of the cluster (only in cluster_mode=True)
def normalization_5(cluster_mode=False, subject=None):
    global inverted_index
    global punctuations

    if not cluster_mode:
        i_i = inverted_index
    else:
        i_i = inverted_index[subject]

    for token in list(i_i.keys()):
        if '.' in token:  # if it is at the end of a line
            reduced_token = token[:-1]  # cloning token
            for p in punctuations:
                if p in token:
                    reduced_token = reduced_token.replace(p, "")
            if reduced_token in i_i.keys() and i_i[reduced_token] is not None:
                i_i[reduced_token] = list(set().union(i_i[reduced_token], i_i[token]))
                i_i.pop(token)
            else:
                i_i[reduced_token] = i_i[token]
                i_i.pop(token)


# removes terms that add no meaning (i.e. "و", "با", "برای", etc.)
# phase 3: cluster mode: determines whether we want an inverted index for all docs, or docs of one particular subject
# phase 3: subject: determines the subject of the cluster (only in cluster_mode=True)
def remove_stop_words(cluster_mode=False, subject=None):
    global inverted_index
    global stop_words

    if not cluster_mode:
        i_i = inverted_index
    else:
        i_i = inverted_index[subject]

    for sw in stop_words:
        i_i.pop(sw, "None")


# returns the list of documents which contains the single word query
# returns an error message if no occurrences were found
# phase 3: cluster mode: determines whether we want an inverted index for all docs, or docs of one particular subject
# phase 3: subject: determines the subject of the cluster (only in cluster_mode=True)
def single_word_query_doc_retriever(query, cluster_mode=False, subject=None):
    global inverted_index

    if not cluster_mode:
        i_i = inverted_index
    else:
        i_i = inverted_index[subject]

    if query not in i_i.keys():
        return "no occurrences were found"
    return i_i[query]


# returns the list of documents which contains the multi word query, sorted in descending order
# phase 3: cluster mode: determines whether we want an inverted index for all docs, or docs of one particular subject
# phase 3: subject: determines the subject of the cluster (only in cluster_mode=True)
def multi_word_query_doc_retriever(query, num_of_docs, cluster_mode=False, subject=None):
    global inverted_index

    if not cluster_mode:
        i_i = inverted_index
    else:
        i_i = inverted_index[subject]

    doc_scores = []
    for i in range(num_of_docs):
        doc_scores.append([i, 0])
    for word in query.split():
        if word in i_i.keys():
            for doc_id in i_i[word]:
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
# phase 3: cluster mode: determines whether we want an inverted index for all docs, or docs of one particular subject
# phase 3: subject: determines the subject of the cluster (only in cluster_mode=True)
# returns this dictionary
def build_champion_lists(tokens, docs, cluster_mode=False, subject=None):
    global inverted_index
    global champion_list
    global doc_lengths

    if not cluster_mode:
        i_i = inverted_index
        c_l = champion_list
        d_l = doc_lengths
    else:
        i_i = inverted_index[subject]
        c_l = champion_list[subject]
        d_l = doc_lengths[subject]

    for token in tokens:
        c_l[token] = {}
        for doc_id in i_i[token].keys():
            c_l[token][doc_id] = i_i[token][doc_id] / d_l[doc_id]
        c_l[token] = sorted(c_l[token].items(), key=lambda kv: kv[1], reverse=True)  # sorting
        # TODO: return best k?


# phase 3: cluster mode: determines whether we want an inverted index for all docs, or docs of one particular subject
# phase 3: subject: determines the subject of the cluster (only in cluster_mode=True)
def compute_doc_length(docs, doc_id, cluster_mode=False, subject=None):
    global inverted_index

    if not cluster_mode:
        i_i = inverted_index
    else:
        i_i = inverted_index[subject]

    len = 0
    for word in docs[doc_id].split():
        len += i_i[word][doc_id] ** 2
    return len ** (1 / 2)


def compute_tf(term, doc):
    term_frequency = 0
    for t in doc.split():
        if term == t:
            term_frequency = term_frequency + 1
    # return term_frequency
    return 1 + math.log10(term_frequency)


# phase 3: cluster mode: determines whether we want an inverted index for all docs, or docs of one particular subject
# phase 3: subject: determines the subject of the cluster (only in cluster_mode=True)
def compute_idf(term, len_of_docs, cluster_mode=False, subject=None):
    global inverted_index

    if not cluster_mode:
        i_i = inverted_index
    else:
        i_i = inverted_index[subject]

    # return len(inverted_index[term])
    return math.log10(len_of_docs / len(i_i[term]))


# phase 3: cluster mode: determines whether we want an inverted index for all docs, or docs of one particular subject
# phase 3: subject: determines the subject of the cluster (only in cluster_mode=True)
def compute_tf_idf(term, doc, len_of_docs, cluster_mode=False, subject=None):
    return compute_tf(term, doc) * compute_idf(term, len_of_docs, cluster_mode, subject)


# phase 3: cluster mode: determines whether we want an inverted index for all docs, or docs of one particular subject
# phase 3: subject: determines the subject of the cluster (only in cluster_mode=True)
def compute_cosine_similarity(query, docs, doc_id, cluster_mode=False, subject=None):
    global inverted_index
    global doc_lengths

    if not cluster_mode:
        i_i = inverted_index
        d_l = doc_lengths
    else:
        i_i = inverted_index[subject]
        d_l = doc_lengths[subject]

    res = 0
    for word in query.split():
        if int(doc_id) in docs.keys():
            if word in docs[int(doc_id)].split():  # casting, just to make sure
                res += compute_tf_idf(word, docs[doc_id], len(docs), cluster_mode, subject) * i_i[word][str(doc_id)]  # casting, just to make sure
    res /= d_l[str(doc_id)]  # casting, just to make sure
    return res


# phase 3: cluster mode: determines whether we want an inverted index for all docs, or docs of one particular subject
# phase 3: subject: determines the subject of the cluster (only in cluster_mode=True)
def compute_scores(query, docs, cluster_mode=False, subject=None):
    global champion_list

    if not cluster_mode:
        c_l = champion_list
    else:
        c_l = champion_list[subject]

    scores = {}
    for word in query.split():
        if word in c_l.keys():
            for doc_tuple in c_l[word]:
                doc_id = int(doc_tuple[0])  # casting, just to make sure
                if doc_id not in scores.keys():
                    scores[doc_id] = 0
                scores[doc_id] += compute_cosine_similarity(word, docs, doc_id, cluster_mode, subject)
    # scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)  # sorting
    return scores


def select_best_k_scores(query, docs, k=3, cluster_mode=False, subject=None):
    scores = compute_scores(query, docs, cluster_mode, subject)
    print('original scores are: {}'.format(scores))
    best_scores = {}
    heap = []
    for doc_id in scores.keys():
        heappush(heap, (-scores[doc_id], doc_id))
    for i in range(k):
        max, best_doc_id = heappop(heap)
        best_scores[best_doc_id] = -max
    return best_scores


# phase 3 functions
def compute_cluster_centroid(cluster_of_docs, subject):
    global inverted_index
    centroid = {}

    print(subject)
    for term in inverted_index[subject].keys():
        tf_sum = sum(inverted_index[subject][term].values())
        n = len(inverted_index[subject][term].keys())
        avg = tf_sum / n
        centroid[term] = avg
    return centroid


# phase 3: cluster mode: determines whether we want an inverted index for all docs, or docs of one particular subject
# phase 3: subject: determines the subject of the cluster (only in cluster_mode=True)
def compute_centroid_length(subject):
    global centroids

    len = 0
    for word in centroids[subject].keys():
        len += centroids[subject][word] ** 2
    return len ** (1 / 2)


# phase 3: cluster mode: determines whether we want an inverted index for all docs, or docs of one particular subject
# phase 3: subject: determines the subject of the cluster (only in cluster_mode=True)
def compute_cosine_similarity_with_centroid(query, doc, len_of_all_docs, centroid_length, cluster_mode, subject):
    global inverted_index
    global centroids

    res = 0
    for word in query.split():
        if word in centroids[subject].keys():
            res += compute_tf_idf(word, doc, len_of_all_docs, cluster_mode, subject) * centroids[subject][word]
    res /= centroid_length
    return res


def find_best_cluster(query, len_of_all_docs):
    global centroids

    max = 0
    best_subject = ""
    for subject in centroids.keys():
        centroids_doc = ' '.join(word for word in centroids[subject].keys())
        centroid_length = compute_centroid_length(subject)
        val = compute_cosine_similarity_with_centroid(query, centroids_doc, len_of_all_docs, centroid_length,
                                                      cluster_mode=True, subject=subject)
        if val > max:
            max = val
            best_subject = subject
    return best_subject, max


def main():
    global inverted_index
    global champion_list
    global centroids
    global doc_lengths

    print("Welcome to my Search Engine!")
    time.sleep(1)

    docs = {}

    # phase 2
    # path = '.\sampleDoc'
    # for dirpath, dirnames, files in os.walk(path):
    #     for i, file_name in enumerate(files):
    #         f = open(path + '\\' + file_name, "r", encoding="utf8")
    #         docs[i] = f.read()  # stores value "doc" in key "i"

    # phase 3
    path = '.\subjects'
    folders = [x[1] for x in os.walk(path)][0]
    num_of_docs_per_folder = 50  # determine the number of docs we want for each folder (subject)
    for folder in folders:
        docs[folder] = {}
        subpath = path + '\\' + folder
        for dirpath, dirnames, files in os.walk(subpath):
            for i, file_name in enumerate(files):
                if i < num_of_docs_per_folder:
                    f = open(subpath + '\\' + file_name, "r", encoding="utf8")
                    file_index = int(file_name[:-4])
                    docs[folder][file_index] = f.read()  # stores value "doc" in key "i"

    # # ---------------------------------------------------------------------------------
    # # pre-computation! (files are already cached. comment this part if you wish!)
    # for subject in docs.keys():
    #     # raw tokens
    #     tokens = set()
    #     for doc_id in docs[subject]:
    #         tokens = tokens.union(tokenize(docs[subject][doc_id]))
    #     # inverted index
    #     inverted_index[subject] = {}
    #     doc_lengths[subject] = {}
    #     build_inverted_index(tokens, docs[subject], cluster_mode=True, subject=subject)
    #     # champion list
    #     champion_list[subject] = {}
    #     build_champion_lists(tokens, docs[subject], cluster_mode=True, subject=subject)
    #
    #     centroids[subject] = compute_cluster_centroid(docs[subject], subject)
    #
    # # # print("-----------------------------------")
    # # # print("inverted index: ")
    # # # print("-----------------------------------")
    # # # time.sleep(1)
    # # # print(inverted_index)
    # #
    # # # time.sleep(0.75)
    # # # print()
    # # # print()
    # # # print()
    # # # time.sleep(0.75)
    # #
    # normalization_1()
    # # # print("-----------------------------------")
    # # # print("inverted index: (after normalization #1)")
    # # # print("-----------------------------------")
    # # # time.sleep(1)
    # # # print(inverted_index)
    # #
    # # # time.sleep(0.75)
    # # # print()
    # # # print()
    # # # print()
    # # # time.sleep(0.75)
    # #
    # normalization_2()
    # # # print("-----------------------------------")
    # # # print("inverted index: (after normalization #2)")
    # # # print("-----------------------------------")
    # # # time.sleep(1)
    # # # print(inverted_index)
    # #
    # # # time.sleep(0.75)
    # # # print()
    # # # print()
    # # # print()
    # # # time.sleep(0.75)
    # #
    # normalization_3()
    # # # print("-----------------------------------")
    # # # print("inverted index: (after normalization #3)")
    # # # print("-----------------------------------")
    # # # time.sleep(1)
    # # # print(inverted_index)
    # #
    # # # time.sleep(0.75)
    # # # print()
    # # # print()
    # # # print()
    # # # time.sleep(0.75)
    # #
    # normalization_4()
    # # # print("-----------------------------------")
    # # # print("inverted index: (after normalization #4)")
    # # # print("-----------------------------------")
    # # # time.sleep(1)
    # # # print(inverted_index)
    # #
    # # # time.sleep(0.75)
    # # # print()
    # # # print()
    # # # print()
    # # # time.sleep(0.75)
    # #
    # # normalization_5()
    # # # print("-----------------------------------")
    # # # print("inverted index: (after normalization #5)")
    # # # print("-----------------------------------")
    # # # time.sleep(1)
    # # # print(inverted_index)
    # #
    # # # time.sleep(0.75)
    # # # print()
    # # # print()
    # # # print()
    # # # time.sleep(0.75)
    # #
    # remove_stop_words()
    # # # print("-----------------------------------")
    # # # print("inverted index: (after eliminating stop words)")
    # # # print("-----------------------------------")
    # # # time.sleep(1)
    # # # print(inverted_index)
    # #
    # # # time.sleep(0.75)
    # # # print()
    # # # print()
    # # # print()
    # # # time.sleep(0.75)
    #
    # # writing inverted_index data into file
    # f = open("inverted_index.txt", "w", encoding="utf8")
    # f.write(json.dumps(inverted_index))
    # f.close()
    #
    # # writing champion_list data into file
    # f = open("champion_list.txt", "w", encoding="utf8")
    # f.write(json.dumps(champion_list))
    # f.close()
    #
    # # writing centroid data into file
    # f = open("centroids.txt", "w", encoding="utf8")
    # f.write(json.dumps(centroids))
    # f.close()
    #
    # # writing doc_lengths data into file
    # f = open("doc_lengths.txt", "w", encoding="utf8")
    # f.write(json.dumps(doc_lengths))
    # f.close()
    # # ---------------------------------------------------------------------------------

    # ---------------------
    # read from files!
    with open("inverted_index.txt", "r") as f1:  # TODO: figure out this shit!
        inverted_index_data = f1.read()
    inverted_index = json.loads(inverted_index_data)

    with open("champion_list.txt", "r") as f2:  # TODO: figure out this shit!
        champion_list_data = f2.read()
    champion_list = json.loads(champion_list_data)

    with open("centroids.txt", "r") as f3:  # TODO: figure out this shit!
        centroids_data = f3.read()
    centroids = json.loads(centroids_data)

    with open("doc_lengths.txt", "r") as f4:  # TODO: figure out this shit!
        doc_lengths_data = f4.read()
    doc_lengths = json.loads(doc_lengths_data)
    # ---------------------

    len_of_all_docs = 0
    for subject in docs.keys():
        len_of_all_docs += len(docs[subject].keys())

    # example queries
    # query = "آشامیدنی"  # health
    # query = "جبری"  # math
    # query = "محاسبات"  # math
    # query = "اشعه" # physics
    query = "شاهنشاهی"  # history
    # query = "سفینه"  # technology
    # query = "سفینه فضایی زحل"  # technology (multi word)
    # query = "اختلال"  # a word that is in multiple subjects
    # query = "اصالت"  # a word that is not in our IR!
    # query = "کوانتوم"
    query = "مشتق انتگرال"
    query = "قلب سنگ پا"

    best_subject, maximum = find_best_cluster(query, len_of_all_docs)
    print('query {} was categorized in subject {} with {} similarity'.format(query, best_subject, maximum))

    if maximum != 0:

        print('best (sorted) scores for query {} are: {}'.format(query, select_best_k_scores(query, docs[best_subject],
                                                                                             k=min(5,
                                                                                                   len(compute_scores(query,
                                                                                                                      docs[
                                                                                                                          best_subject],
                                                                                                                      cluster_mode=True,
                                                                                                                      subject=best_subject).keys())),
                                                                                             cluster_mode=True,
                                                                                             subject=best_subject)))


if __name__ == "__main__":
    main()
