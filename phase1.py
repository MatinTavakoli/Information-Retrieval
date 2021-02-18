import time
import os

inverted_index = {}
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


# makes a dictionary, where for every token, an array of doc_ids are is created
# returns this dictionary
def build_inverted_index(tokens, docs):
    global inverted_index

    for token in tokens:
        inverted_index[token] = []
        for doc_id in docs:
            if token in docs[doc_id].split():
                inverted_index[token].append(doc_id)
    return inverted_index


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
        reduced_token = (token + '.')[:-1]  # cloning token
        for p in punctuations:
            if p in token:
                reduced_token = reduced_token.replace(p, "")
        if reduced_token in inverted_index.keys():
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
    postings = inverted_index[query]
    return postings


# returns the list of documents which contains the multi word query, sorted in descending order
def multi_word_query_doc_retriever(query):
    global inverted_index
    doc_scores = {}
    for word in query.split():
        if word in inverted_index.keys():
            for doc_id in inverted_index[word]:
                if doc_id not in doc_scores.keys():
                    doc_scores[doc_id] = 1
                else:
                    doc_scores[doc_id] = doc_scores[doc_id] + 1

    doc_scores = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)  # sorting

    filtered_doc_scores = [(id, score) for (id, score) in doc_scores if score != 0]  # filtering the zeros out

    return filtered_doc_scores


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
    for doc in docs.keys():
        tokens = tokens.union(tokenize(docs[doc]))
    # print("-----------------------------------")
    # print("raw tokens: ")
    # print("-----------------------------------")
    # time.sleep(1)
    # print(tokens)
    #
    # time.sleep(0.75)
    # print()
    # print()
    # print()
    # time.sleep(0.75)
    #
    # # inverted index
    build_inverted_index(tokens, docs)
    # print("-----------------------------------")
    # print("inverted index: ")
    # print("-----------------------------------")
    # time.sleep(1)
    # print(inverted_index)
    # # print(inverted_index["آغاز"])
    # # print(inverted_index["بارسلونا"])
    # # print(inverted_index["کرد"])
    # # print(inverted_index["و"])
    #
    # time.sleep(0.75)
    # print()
    # print()
    # print()
    # time.sleep(0.75)
    #
    # print("-----------------------------------")
    # print("inverted index: (after normalization #1)")
    # print("-----------------------------------")
    # time.sleep(1)
    normalization_1()
    # print(inverted_index)
    #
    # time.sleep(0.75)
    # print()
    # print()
    # print()
    # time.sleep(0.75)
    #
    # print("-----------------------------------")
    # print("inverted index: (after normalization #2)")
    # print("-----------------------------------")
    # time.sleep(1)
    normalization_2()
    # print(inverted_index)
    #
    # time.sleep(0.75)
    # print()
    # print()
    # print()
    # time.sleep(0.75)
    #
    # print("-----------------------------------")
    # print("inverted index: (after normalization #3)")
    # print("-----------------------------------")
    # time.sleep(1)
    normalization_3()
    # print(inverted_index)
    #
    # time.sleep(0.75)
    # print()
    # print()
    # print()
    # time.sleep(0.75)
    #
    # print("-----------------------------------")
    # print("inverted index: (after normalization #4)")
    # print("-----------------------------------")
    # time.sleep(1)
    normalization_4()
    # print(inverted_index)
    #
    # time.sleep(0.75)
    # print()
    # print()
    # print()
    # time.sleep(0.75)
    #
    # print("-----------------------------------")
    # print("inverted index: (after normalization #5)")
    # print("-----------------------------------")
    # time.sleep(1)
    # normalization_5()
    #
    # print(inverted_index)
    #
    # time.sleep(0.75)
    # print()
    # print()
    # print()
    # time.sleep(0.75)
    #
    remove_stop_words()
    # print("-----------------------------------")
    # print("inverted index: (after eliminating stop words)")
    # print("-----------------------------------")
    # time.sleep(1)
    # print(inverted_index)
    #
    # time.sleep(0.75)
    # print()
    # print()
    # print()
    # time.sleep(0.75)

    # testing queries (single word)
    print("-----------------------------------")
    print("testing search engine model (on single word queries)")
    print("-----------------------------------")
    time.sleep(1)
    # print(single_word_query_doc_retriever("ایران"))
    # print(single_word_query_doc_retriever("پروژه"))
    # print(single_word_query_doc_retriever("کاربر"))
    # print(single_word_query_doc_retriever("کاربران"))
    print(single_word_query_doc_retriever("الهامی"))
    # print(single_word_query_doc_retriever("فجر"))
    # print(single_word_query_doc_retriever("بهمن"))
    # print(single_word_query_doc_retriever("سینما"))
    # print(single_word_query_doc_retriever("لیگ"))
    # print(single_word_query_doc_retriever("قهرمانان"))
    # print(single_word_query_doc_retriever("آسیا"))
    # print(single_word_query_doc_retriever("استغفار"))
    # print('%%%')
    # print(single_word_query_doc_retriever("به"))
    # print(single_word_query_doc_retriever("با"))
    # print(single_word_query_doc_retriever("برای"))
    # print('%%%')

    time.sleep(0.75)
    print()
    print()
    print()
    time.sleep(0.75)

    # testing queries (multi word)
    print("-----------------------------------")
    print("testing search engine model (on multi word queries)")
    print("-----------------------------------")
    time.sleep(1)
    # print(multi_word_query_doc_retriever("رقابت ورزشی"))
    # print(multi_word_query_doc_retriever("گذشته ندید"))
    # print(multi_word_query_doc_retriever("فجر بهمن سینما"))
    # print(multi_word_query_doc_retriever("لیگ قهرمانان آسیا"))


if __name__ == "__main__":
    main()
