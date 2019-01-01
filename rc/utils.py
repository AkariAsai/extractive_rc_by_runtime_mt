import re
import numpy as np
import csv
import os
import json
import unicodedata
from tqdm import tqdm
import MeCab
# To use google translate, need to load api_key in .env file.
from dotenv import load_dotenv
from os.path import join, dirname
from pycorenlp import StanfordCoreNLP
import requests
import os
import uuid, json
import subprocess

# TODO: TO experiment on Japanese SQuAD dataset, you first need to install mecab-python3 `<https://pypi.org/project/mecab-python3/>`_.
# Also, you need to install mecab-ipadic-NEologd `<https://github.com/neologd/mecab-ipadic-neologd/blob/master/README.md>`_. 
tagger = MeCab.Tagger('-d /home/asai/local/lib/mecab/dic/mecab-ipadic-neologd')
nlp_server = StanfordCoreNLP('http://localhost:10000')

def get_2d_spans(text, tokenss):
    spanss = []
    cur_idx = 0
    for tokens in tokenss:
        spans = []
        for token in tokens:
            if text.find(token, cur_idx) < 0:
                print(tokens)
                print("{} {} {}".format(token, cur_idx, text))
                raise Exception()
            cur_idx = text.find(token, cur_idx)
            spans.append((cur_idx, cur_idx + len(token)))
            cur_idx += len(token)
        spanss.append(spans)
    return spanss


def get_word_span(context, wordss, start, stop):
    spanss = get_2d_spans(context, wordss)
    idxs = []
    for sent_idx, spans in enumerate(spanss):
        for word_idx, span in enumerate(spans):
            if not (stop <= span[0] or start >= span[1]):
                idxs.append((sent_idx, word_idx))

    assert len(idxs) > 0, "{} {} {} {}".format(context, spanss, start, stop)
    return idxs[0], (idxs[-1][0], idxs[-1][1] + 1)


def get_phrase(context, wordss, span):
    start, stop = span
    flat_start = get_flat_idx(wordss, start)
    flat_stop = get_flat_idx(wordss, stop)
    words = sum(wordss, [])
    char_idx = 0
    char_start, char_stop = None, None
    for word_idx, word in enumerate(words):
        char_idx = context.find(word, char_idx)
        assert char_idx >= 0
        if word_idx == flat_start:
            char_start = char_idx
        char_idx += len(word)
        if word_idx == flat_stop - 1:
            char_stop = char_idx
    assert char_start is not None
    assert char_stop is not None
    return context[char_start:char_stop]


def get_flat_idx(wordss, idx):
    return sum(len(words) for words in wordss[:idx[0]]) + idx[1]


def get_word_idx(context, wordss, idx):
    spanss = get_2d_spans(context, wordss)
    return spanss[idx[0]][idx[1]][0]


def get_attention_matrix_dic(attention_context_filepath):
    '''
    This is a function for making attention array dictionary.
    The output is a ditionary like
    {global_sent_idx : [[0.3 ,.., 0.4], [...], ..., [...], ]}
    attn_dic[global_sent_idx][local_token_idx] will returns an vector
    whose length would be the len(source_lines[global_sent_idx])
    and the argmax(array) will returns the token with the highest attention weight
    in the row.
    '''
    attn_f = open(attention_context_filepath, "r")
    data = attn_f.read()
    data = data.split("\n")
    attn_dic = {}
    sent_id = 0

    for line in data:
        if len(line) == 0:
            sent_id += 1
        else:
            array = [float(num) for num in line.split(" ")]
            attn_dic.setdefault(sent_id, [])
            if sent_id in attn_dic:
                attn_dic[sent_id].append(array)
    for k, v in attn_dic.items():
        attn_dic[k] = v[:-1]

    return attn_dic


def get_attention_indeices(attention_context_filepath):
    '''
    This is a function for making attention array dictionary.
    The output is a ditionary like
    {global_sent_idx : [[0.3 ,.., 0.4], [...], ..., [...], ]}
    attn_dic[global_sent_idx][local_token_idx] will returns an vector
    whose length would be the len(source_lines[global_sent_idx])
    and the argmax(array) will returns the token with the highest attention weight
    in the row.
    '''
    attn_f = open(attention_context_filepath, "r")
    data = attn_f.read().splitlines()
    attn_indices_dic = {}
    sent_id = 0

    for i, attn_indices in enumerate(data):
        attn_indices_dic[i] = [int(index) for index in attn_indices.split()]

    return attn_indices_dic


def get_title2attention_dic(attention_matrix, sent_idx_dic):
    title2attention_dic = {}
    for title in sent_idx_dic.keys():
        title2attention_dic.setdefault(title, {})
        for para_idx in sent_idx_dic[title].keys():
            sent_indices = sent_idx_dic[title][para_idx]
            title2attention_dic[title][para_idx] = [
                attention_matrix[index] for index in sent_indices]
    return title2attention_dic


def get_sentence_idx_word_index_in_sentence(index_in_paragraph, paragraph):
    print(index_in_paragraph)
    sent_index = 0
    word_index = 0
    tokenized_paragraph = paragraph.split()
    print(paragraph)

    for i in range(index_in_paragraph):
        if tokenized_paragraph[i] == ".":
            sent_index += 1
            word_index = 0
        else:
            word_index += 1
    print("index in paragraph is : {0}, sent index:{1}, word_index:{2}".format(
        index_in_paragraph, sent_index, word_index))
    return sent_index, word_index

def get_source_lines_trans_lines(source_context_filename, trans_context_filename):
    source_lines = []
    with open(source_context_filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip the header
        for row in reader:
            source_lines.append(row[3])
    target_f = open(trans_context_filename)
    target_lines = target_f.readlines()
    assert len(source_lines) == len(target_lines)

    return source_lines, target_lines


def get_global_sent_idx_local_token_idx(titile, paragraph_id, global_token_idx,
                                        para_sent_dict, target_lines):
    '''
    This is a function to get global token idx.
    Unfortunately for the memory size limitations, we stores the token sequence without
    sentence segmentation.
    we call the index returned from model calls global_token_idx,
    ["The", "dog", "is", "my", "dog", ".", "She", "is", "really", "cute", "."]
    Here, the "dog" would be represent (0, 1) and "She" is (0, 6)
    '''
    sentences = []
    for num in para_sent_dict[titile][paragraph_id]:
        sentences.append(target_lines[num])
    # Remove the new line
    sentences = [sentence.split(" ")[:-1] for sentence in sentences]

    global_sent_idx = 0
    local_token_idx = 0
    prev = 0
    for (i, sentence) in enumerate(sentences):
        if prev + len(sentence) <= global_token_idx:
            prev += len(sentence)
        else:
            global_sent_idx = para_sent_dict[titile][paragraph_id][i]
            local_token_idx = global_token_idx - prev
            break
    return global_sent_idx, local_token_idx


def get_highest_attention_source_idx(attention_dic, sentence_idx, word_idx):
    return attention_dic[sentence_idx][word_idx]


def process_tokens(temp_tokens):
    tokens = []
    for token in temp_tokens:
        flag = False
        l = ("\u2212", "\u2014", "\u2013", "/", "~",
             "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        # \u2013 is en-dash. Used for number to nubmer
        # l = ("-", "\u2212", "\u2014", "\u2013")
        # l = ("\u2013",)
        tokens.extend(re.split("([{}])".format("".join(l)), token))
    return tokens


def get_best_span(ypi, yp2i, window_size=None):
    max_val = 0
    best_word_span = (0, 1)
    best_sent_idx = 0
    for f, (ypif, yp2if) in enumerate(zip(ypi, yp2i)):
        argmax_j1 = 0
        if window_size == None:
            for j in range(len(ypif)):
                val1 = ypif[argmax_j1]
                if val1 < ypif[j]:
                    val1 = ypif[j]
                    argmax_j1 = j

                val2 = yp2if[j]
                if val1 * val2 > max_val:
                    best_word_span = (argmax_j1, j)
                    best_sent_idx = f
                    max_val = val1 * val2
        else:
            for j in range(min(len(ypif), window_size)):
                val1 = ypif[argmax_j1]
                if val1 < ypif[j]:
                    val1 = ypif[j]
                    argmax_j1 = j

                val2 = yp2if[j]
                if val1 * val2 > max_val:
                    best_word_span = (argmax_j1, j)
                    best_sent_idx = f
                    max_val = val1 * val2

    return ((best_sent_idx, best_word_span[0]), (best_sent_idx, best_word_span[1] + 1)), float(max_val)


def get_k_best_span(ypi, yp2i, k=5, window_size=None):
    # Track the answer boundary with score.
    '''
    The result would be
    [[(start_sent_index, start_word_postion), (end_sent_index, end_word_position), score],
    [(start_sent_index, start_word_postion),
      (end_sent_index, end_word_position), score]
    ...] for k
    The first one should have the highest score.
    '''

    score_dic = {}
    position_dic = {}

    ans_idx = 0
    for f, (ypif, yp2if) in enumerate(zip(ypi, yp2i)):
        argmax_j1 = 0
        if window_size == None:
            for j in range(len(ypif)):
                val1 = ypif[argmax_j1]
                if val1 < ypif[j]:
                    val1 = ypif[j]
                    argmax_j1 = j

                val2 = yp2if[j]
                score_dic[ans_idx] = val1 * val2
                position_dic[ans_idx] = [(f, argmax_j1), (f, j + 1)]
                ans_idx += 1
        else:
            for j in range(min(len(ypif), window_size)):
                val1 = ypif[argmax_j1]
                if val1 < ypif[j]:
                    val1 = ypif[j]
                    argmax_j1 = j

                val2 = yp2if[j]
                score_dic[ans_idx] = val1 * val2
                position_dic[ans_idx] = [(f, argmax_j1), (f, j + 1)]
                ans_idx += 1

    sorted_id = [k for k in sorted(
        score_dic, key=score_dic.get, reverse=True)]

    k_best = []
    for i in sorted_id[:5]:
        k_best.append([position_dic[i][0], position_dic[i][1], score_dic[i]])

    return k_best


def get_best_span_wy(wypi, th):
    chunk_spans = []
    scores = []
    chunk_start = None
    score = 0
    l = 0
    th = min(th, np.max(wypi))
    for f, wypif in enumerate(wypi):
        for j, wypifj in enumerate(wypif):
            if wypifj >= th:
                if chunk_start is None:
                    chunk_start = f, j
                score += wypifj
                l += 1
            else:
                if chunk_start is not None:
                    chunk_stop = f, j
                    chunk_spans.append((chunk_start, chunk_stop))
                    scores.append(score / l)
                    score = 0
                    l = 0
                    chunk_start = None
        if chunk_start is not None:
            chunk_stop = f, j + 1
            chunk_spans.append((chunk_start, chunk_stop))
            scores.append(score / l)
            score = 0
            l = 0
            chunk_start = None

    return max(zip(chunk_spans, scores), key=lambda pair: pair[1])


def get_span_score_pairs(ypi, yp2i):
    span_score_pairs = []
    for f, (ypif, yp2if) in enumerate(zip(ypi, yp2i)):
        for j in range(len(ypif)):
            for k in range(j, len(yp2if)):
                span = ((f, j), (f, k + 1))
                score = ypif[j] * yp2if[k]
                span_score_pairs.append((span, score))
    return span_score_pairs


# Code for Japanese text tokenization and reguralization.
def three_digits(text):
    # This is a method to replace 1 , 345 , 678 for 1,345,678.
    text_mod = \
        re.sub('([0-9]+) (,) ([0-9]{3})', r'\1\2\3', text)
    text_mod = \
        re.sub('([0-9]+) (,) ([0-9]{3})', r'\1\2\3', text_mod)
    return text_mod


def add_space_before_units(text):
    text = re.sub('([0-9]+)(人)', r'\1 \2', text)
    text = re.sub('([0-9]+)(年)', r'\1 \2', text)
    text = re.sub('([0-9]+)(歳)', r'\1 \2', text)
    text = re.sub('([0-9]+)(年間)', r'\1 \2', text)
    text = re.sub('([0-9]+)(世紀)', r'\1 \2', text)
    text = re.sub('([0-9]+)(km)', r'\1 \2', text)
    text = re.sub('([0-9]+)(cm)', r'\1 \2', text)
    text = re.sub('([0-9]+)(m)', r'\1 \2', text)
    return text


def add_space_before_date(text):
    text = re.sub('([0-9]+)(年)([0-9]+)(月)([0-9]+)(日)',
                  r'\1\2 \3\4 \5\6', text)
    text = re.sub('([0-9]+)(月)([0-9]+)(日)', r'\1\2 \3\4', text)
    return text


def remove_space_before_decimal_point(text):
    text = re.sub('([0-9]+) (\.) ([0-9]+)', r'\1\2\3', text)
    return text


def convert_num_alpha_to_half_size(text):
    text = unicodedata.normalize("NFKC", text)
    return text


def convert_result_to_tokenized_sentence(output_json):
    if type(output_json) == str or len(output_json['sentences']) == 0 or len(output_json['sentences'][0]) < 1:
        return ""
    result = output_json['sentences'][0]['tokens']
    tokens = []
    for token in result:
        distinct_words_en.setdefault(token['originalText'], 0)
        distinct_words_en[token['originalText']] += 1

        tokens.append(token['originalText'])
    return " ".join(tokens)

def japanese_preprocessing(sentence):
    text = sentence.rstrip()
    text = text.replace("\u3000", "")
    text = three_digits(text)
    text = add_space_before_date(text)
    text = add_space_before_units(text)
    text = remove_space_before_decimal_point(text)
    text = convert_num_alpha_to_half_size(text)
    return text


def extract_word(sentence):
    words = []
    sentence = japanese_preprocessing(sentence)

    for chunk in tagger.parse(sentence).splitlines()[:-1]:
        result = chunk.split('\t')
        if len(result) < 2 or len(result) > 2:
            return ""
        surface = result[0]
        words.append(surface)
    return " ".join(words)


def tokenize_preprocess_japanese_sent(sentence):
    words = []
    # First remvove new line and all-space, and convert numalphas to half
    # spaces
    sentence = remove_new_line(sentence)
    sentence = sentence.replace("\u3000", "")
    sentence = convert_num_alpha_to_half_size(sentence)

    for chunk in tagger.parse(sentence).splitlines()[:-1]:
        result = chunk.split('\t')
        # surface = result[0]
        surface = result[0].lower()
        words.append(surface)

    sentence = " ".join(words)
    sentence = japanese_preprocessing(sentence)

    return sentence


# Functions for tokenize English using Google Translate.
def remove_new_line(sentence):
    return sentence.rstrip()


def convert_result_to_tokenized_sentence(output_json):
    if type(output_json) == str or len(output_json['sentences']) == 0 or len(output_json['sentences'][0]) < 1:
        return ""
    result = output_json['sentences'][0]['tokens']
    tokens = []
    for token in result:
        tokens.append(token['originalText'])
    tokens = [token.lower() for token in tokens]
    return " ".join(tokens)


def tokenize_preprocess_english_sent(sentence, is_lower=True):
    if type(sentence) != str:
        return ""
    sentence = remove_new_line(sentence)

    if is_lower:
        sentence = sentence.lower()
    output = nlp_server.annotate(sentence,
                                 properties={'annotators': 'tokenize, ssplit', 'outputFormat': 'json'})
    return convert_result_to_tokenized_sentence(output)

# Functions to use Google translate
# To use Google Translate to evaluate the baseline performance, 
# you first need to get your own API key, and add key to .env file.
def google_translate(sentence, toJa, lang=None):
    print("google_trans has been callsed.")
    dotenv_path = join(dirname(__file__), '.env')
    load_dotenv(dotenv_path)

    api_key = os.environ.get("API_KEY")
    url = "https://translation.googleapis.com/language/translate/v2"
    url += "?key=" + api_key
    url += "&q=" + sentence
    if lang:
        url += "&source=" + lang + "&target=en"
    elif toJa == True:
        url += "&source=en&target=ja"
    else:
        url += "&source=ja&target=en"

    rr = requests.get(url)

    unit_aa = json.loads(rr.text)
    # print(unit_aa)
    if 'error' in unit_aa:
        return 'BAD REQUEST'
    else:
        result = unit_aa["data"]["translations"][0]["translatedText"]
        return result


def google_translate_to_fr(sentence, toFr, lang=None):
    print("google_trans has been callsed.")
    dotenv_path = join(dirname(__file__), '.env')
    load_dotenv(dotenv_path)

    api_key = os.environ.get("API_KEY")
    url = "https://translation.googleapis.com/language/translate/v2"
    url += "?key=" + api_key
    url += "&q=" + sentence
    if lang:
        url += "&source=" + lang + "&target=en"
    elif toFr == True:
        url += "&source=en&target=fr"
    else:
        url += "&source=fr&target=en"

    rr = requests.get(url)

    unit_aa = json.loads(rr.text)
    # print(unit_aa)
    if 'error' in unit_aa:
        return 'BAD REQUEST'
    else:
        result = unit_aa["data"]["translations"][0]["translatedText"]
        return result

# TODO: You need to fix some paths. 
def evaluate_bleu_scores_google_translate(source_file, ground_truth, lang):
    source_f = open(source_file, "r")
    sources = source_f.readlines()
    translated_fp = "google_translated_" + lang + ".txt"
    translated_f = open(translated_fp, "w")
    for line in lines:
        result = google_translate(line, False, lang)
        result = tokenize_preprocess_english_sent(result)
        translated_f.write(result + "\n")
    translated_f.close()
    os.sys("perl nmt/tools/multi-bleu.perl " + ground_truth +
           " < " + translated_fp + " > bleu.txt")

def evaluate_bleu_scores_bing_translate(source_file, ground_truth, from_lang, to_lang):
    source_f = open(source_file, "r")
    sources = source_f.readlines()
    translated_fp = "bing_translated_from_{0}_to_{1}.txt".format(from_lang, to_lang)
    translated_f = open(translated_fp, "w")
    for line in tqdm(sources):
        result = bing_translate(line, from_lang, to_lang)
        result = tokenize_preprocess_english_sent(result)
        translated_f.write(result + "\n")
    translated_f.close()
    os.sys("perl nmt/tools/multi-bleu.perl " + ground_truth +
           " < " + translated_fp + " > bleu.txt")

def normalize_tokenized_sent(filepath, lang):
    os.system("../mosesdecoder-master/scripts/tokenizer/normalize-punctuation.perl < " +
              filepath + " > normalized.txt")
    os.system("../mosesdecoder-master/scripts/tokenizer/tokenizer.perl -l " +
              lang.lower() + " < normalized.txt > tmp.txt")

    normalized_text = open("tmp.txt","r").readlines()
    result_f = open(filepath, "w")
    for text in normalized_text:
        print(text.replace("&apos;", "'").replace("&quot;", '"').replace("&amp;", "&"))
        result_f.write(text.replace("&apos;", "'").replace("&quot;", '"').replace("&amp;", "&"))
    result_f.close()

def normalize_tokenized_answers(ans_1, ans_2, ans_3, lang):
    # You need to fix moses to escape special chars
    temp_f = open(
        "../mosesdecoder-master/scripts/tokenizer/fr_ans_temp.txt", "w")
    temp_f.write(ans_1 + "\n")
    temp_f.write(ans_2 + "\n")
    temp_f.write(ans_3 + "\n")
    temp_f.close()

    os.system("../mosesdecoder-master/scripts/tokenizer/normalize-punctuation.perl < ../mosesdecoder-master/scripts/tokenizer/fr_ans_temp.txt > ../mosesdecoder-master/scripts/tokenizer/normalized.txt")
    os.system("../mosesdecoder-master/scripts/tokenizer/tokenizer.perl -l " +
              lang.lower() + " < ../mosesdecoder-master/scripts/tokenizer/normalized.txt > ../mosesdecoder-master/scripts/tokenizer/tokenized_ans.txt")
    tokenized_f = open(
        "../mosesdecoder-master/scripts/tokenizer/tokenized_ans.txt", "r")
    tokenized_ans = tokenized_f.readlines()
    anss = [ans.rstrip().lower().replace("&quot;",'"').replace("&apos;", "'") for ans in tokenized_ans]
    print(anss)

    return anss

def translate_with_bing(sentence, target_lang="ja", return_align=False):
    dotenv_path = join(dirname(__file__), '.env')
    load_dotenv(dotenv_path)

    subscriptionKey = os.environ.get("API_KEY_BING")

    base_url = 'https://api.cognitive.microsofttranslator.com'
    path = '/translate?api-version=3.0'
    params = '&from=en&to='+ target_lang +'&includeAlignment=true'
    constructed_url = base_url + path + params

    headers = {
        'Ocp-Apim-Subscription-Key': subscriptionKey,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    body = [{
        'text' : sentence
    }]

    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()
    
    if return_align == True:
        return response[0]['translations'][0]['text'],response[0]['translations'][0]['alignment']
    
    return response[0]['translations'][0]['text']

def convert_index_alignment_dict(align_string):
    align_info = align_string.split()
    align_dict = {}
    for item in align_info:
        source, target = item.split("-")
        source = source.split(":")
        target = target.split(":")
        
        align_dict[(int(target[0]), int(target[1]))] = \
            (int(source[0]), int(source[1]))
        
    return align_dict

def bing_based_answer_retreival(answer, source_sent, align_string):
    """
    inputs:
    answer: string (in the source language, English)
    souorce_sent:string (in the target language)
    align_string: string, the alignment projection info from bing translator
    """
    align_dict = convert_index_alignment_dict(align_string)
    predicted_index = \
        [source_sent.find(answer), source_sent.find(answer)+len(answer)]

    orig_ans = ""
    for k,v in align_dict.items():
        if k[0] >= predicted_index[0] and k[1] <= predicted_index[1]:
            orig_ans += ja[v[0]:v[1]+1]

    return orig_ans

# Easy to use function for Bing Translator
def bing_translate(sentence, from_lang, to_lang, return_align=False):
    dotenv_path = join(dirname(__file__), '.env')
    load_dotenv(dotenv_path)
    subscriptionKey = os.environ.get("API_KEY_BING")

    base_url = 'https://api.cognitive.microsofttranslator.com'
    path = '/translate?api-version=3.0'
    params = '&from={0}&to={1}&includeAlignment=true'.format(from_lang, to_lang)
    constructed_url = base_url + path + params

    headers = {
        'Ocp-Apim-Subscription-Key': subscriptionKey,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    body = [{
        'text' : sentence
    }]

    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()

    if return_align == True:
        return response[0]['translations'][0]['text'],response[0]['translations'][0]['alignment']['proj']
    
    return response[0]['translations'][0]['text']

def create_sent_idx_dic(source_context_file_path):
    sent_idx_dic = {}
    with open(source_context_file_path,  newline='') as f:
        dataReader = csv.reader(f)
        header = next(dataReader)
        for row in dataReader:
            sent_idx, title, para_idx, source_sentence = int(
                row[0]), row[1], int(row[2]), row[3]
            sent_idx_dic.setdefault(title, {})
            sent_idx_dic[title].setdefault(para_idx, [])
            sent_idx_dic[title][para_idx].append(sent_idx)

    return sent_idx_dic

def create_questionid2title_dic(question_file_path):
    question_id2title = {}
    with open(question_file_path,  newline='') as f:
        dataReader = csv.reader(f)
        header = next(dataReader)
        for row in dataReader:
            question_id, title, para_idx = int(row[0]), row[1], int(row[2])
            question_id2title[question_id] = {
                "title": title, "para_idx": para_idx}
    return question_id2title
