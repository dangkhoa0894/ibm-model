from nltk.tokenize import word_tokenize
from nltk.translate.ibm1 import IBMModel1
from nltk.translate.ibm2 import IBMModel2
from nltk.translate.ibm3 import IBMModel3
from nltk.translate import AlignedSent
import numpy as np
import pandas

def nltk_ibm_one(data, iter):
    dual_text = []
    translation_prob = {}
    all_german = set()
    all_english = set()
    for d_i in range(len(data)):
        for ge_word in word_tokenize(data[d_i]['dest']):
            all_german.add(ge_word)
        for en_word in word_tokenize(data[d_i]['src']):
            all_english.add(en_word)
    dual_text = []
    for d_i in range(len(data)):
        fr_sent = word_tokenize(data[d_i]['dest'])
        eng_sent = word_tokenize(data[d_i]['src'])
        dual_text.append(AlignedSent(fr_sent, eng_sent))

    ibm_one = IBMModel1(dual_text, iter)
    for german_word in all_german:
        translation_prob[german_word] = {}
        for eng_word in all_english:
             translation_prob[german_word][eng_word] = ibm_one.translation_table[german_word][eng_word]
    translation_prob = pandas.DataFrame.from_dict(translation_prob)
    return translation_prob
    

def nltk_ibm_two(data, iter):
    dual_text = []
    translation_prob = {}
    all_german = set()
    all_english = set()
    for d_i in range(len(data)):
        for ge_word in word_tokenize(data[d_i]['dest']):
            all_german.add(ge_word)
        for en_word in word_tokenize(data[d_i]['src']):
            all_english.add(en_word)
    dual_text = []
    for d_i in range(len(data)):
        fr_sent = word_tokenize(data[d_i]['dest'])
        eng_sent = word_tokenize(data[d_i]['src'])
        dual_text.append(AlignedSent(fr_sent, eng_sent))

    ibm_two = IBMModel2(dual_text, iter)
    for german_word in all_german:
        translation_prob[german_word] = {}
        for eng_word in all_english:
             translation_prob[german_word][eng_word] = ibm_two.translation_table[german_word][eng_word]
    translation_prob = pandas.DataFrame.from_dict(translation_prob)
    return translation_prob

def nltk_ibm_three(data, iter):
    dual_text = []
    translation_prob = {}
    all_german = set()
    all_english = set()
    for d_i in range(len(data)):
        for ge_word in word_tokenize(data[d_i]['dest']):
            all_german.add(ge_word)
        for en_word in word_tokenize(data[d_i]['src']):
            all_english.add(en_word)
    dual_text = []
    for d_i in range(len(data)):
        fr_sent = word_tokenize(data[d_i]['dest'])
        eng_sent = word_tokenize(data[d_i]['src'])
        dual_text.append(AlignedSent(fr_sent, eng_sent))

    ibm_three = IBMModel3(dual_text, iter)
    for german_word in all_german:
        translation_prob[german_word] = {}
        for eng_word in all_english:
             translation_prob[german_word][eng_word] = ibm_three.translation_table[german_word][eng_word]
    translation_prob = pandas.DataFrame.from_dict(translation_prob)
    return translation_prob
    