import numpy as np
import jieba
import random
import re


def create_vocab(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def text_wrangle(text):
    line = re.sub(r'[a-zA-z.【】0-9、。，/！...~*\n]', '', text)
    list_tokens = jieba.cut(line, cut_all=False)
    return [tok.lower() for tok in list_tokens if len(tok) > 1]


def set_word2vec(vocab, doc_list):
    res = [0] * len(vocab)
    for index, word in enumerate(vocab):
        if word in doc_list:
            res[index] = 1
    return res


def train(matrix, label):
    train_nums = len(matrix)  # 样本的个数
    word_nums = len(matrix[0])  # 每个文档的词数（字典长度）
    train_matrix = np.array(matrix)
    train_category = np.array(label)

    pB = sum(train_category) / train_nums
    pA = 1 - pB

    pa = np.ones(word_nums)
    pb = np.ones(word_nums)
    pa_ = 2.0
    pb_ = 2.0
    for i in range(train_nums):
        if train_category[i] == 1:
            pb += train_matrix[i]
            pb_ += sum(train_matrix[i])
        else:
            pa += train_matrix[i]
            pa_ += sum(train_matrix[i])

    pa = np.log(pa / pa_)
    pb = np.log(pb / pb_)

    return pa, pb, pA


def classifyBN(inputMatrix, pa, pb, pB):
    p1 = sum(inputMatrix * pa) + np.log(1 - pB)
    p2 = sum(inputMatrix * pb) + np.log(pB)
    if p1 > p2:
        return 0
    else:
        return 1


def classify():
    doc_list = []
    class_list = []
    with open('./datasets/spam.txt', mode='r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            line = text_wrangle(line)
            doc_list.append(line)
            class_list.append(1)

    with open('./datasets/ham.txt', mode='r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            line = text_wrangle(line)
            doc_list.append(line)
            class_list.append(0)

    vocab = create_vocab(doc_list)

    train_set = list(range(len(doc_list)))
    test_set = []

    for _ in range(int(len(doc_list) * 0.25)):
        rand_index = int(random.uniform(0, len(train_set)))
        test_set.append(rand_index)
        del (train_set[rand_index])

    train_mat = []
    train_class = []
    for i in train_set:
        train_mat.append(set_word2vec(vocab, doc_list[i]))
        train_class.append(class_list[i])

    pa, pb, pA = train(train_mat, train_class)

    count_error = 0
    for i in test_set:
        wordVec = set_word2vec(vocab, doc_list[i])
        result = classifyBN(wordVec, pa, pb, pA)
        print('*' * 20)
        print('prediction results:', result)
        print('true results ', class_list[i])
        print('*' * 20)
        if result != class_list[i]:
            count_error += 1

    print(count_error)


if __name__ == '__main__':
    classify()
