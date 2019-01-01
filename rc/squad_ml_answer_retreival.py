import numpy as np
import os
import ntpath
import json

from utils import get_attention_matrix_dic, \
    get_source_lines_trans_lines, get_global_sent_idx_local_token_idx, \
    get_title2attention_dic, get_sentence_idx_word_index_in_sentence, tokenize_preprocess_japanese_sent, \
    get_attention_indeices, create_questionid2title_dic, create_sent_idx_dic

class SQuADMLAnswerRetreival(object):
    def __init__(self, lang, version=3, embedding_name='embedding', beam_search=False, soft=False):
        self.beam_search = beam_search
        self.soft = soft

        self.set_question_context_answer_file(lang, version, embedding_name)
        self.sent_idx_dic = create_sent_idx_dic(
            self.source_context_file_path)

        if self.beam_search == True and self.soft == False:
            self.attention_matrix = get_attention_indeices(
                self.context_attention_file_path)
        else:
            self.attention_matrix = get_attention_matrix_dic(
                self.context_attention_file_path)
        self.title2attention_dic = get_title2attention_dic(
            self.attention_matrix, self.sent_idx_dic)
        self.question_id2title = create_questionid2title_dic(
            self.question_file_path)
        self.source_lines, self.trans_lines = \
            get_source_lines_trans_lines(
                self.source_context_file_path, self.trans_context_file_path)

        if self.lang == "Fr" or self.lang == "De":
            self.source_lines = open(
                self.japanese_context_file_path, "r").readlines()
            self.source_lines = [sentence.rstrip()
                                 for sentence in self.source_lines]

    def get_id2answerindices_dict(self, answer_spans, question_ids):
        id2answerindices_dict = {int(id_): span
                                 for id_, span in
                                 zip(question_ids, answer_spans)}
        return id2answerindices_dict

    def set_question_context_answer_file(self, lang, version, embedding_name):
        if lang == "Fr":
            if not os.path.exists("trans_result/fr"):
                os.makedirs("trans_result/fr")

            if not os.path.exists("trans_result/fr/v5"):
                os.makedirs("trans_result/fr/v5")
            self.question_file_path = \
                "../data/fr_question_v5.csv"
            self.source_context_file_path = \
                "../data/fr_question_v5_context.csv"
            trans_result_dir_name = \
                os.path.join("trans_result/fr/v5", embedding_name)

            # For french, you need to you pre-tokenized file.
            self.japanese_context_file_path = '../nmt/french_context_tmp.txt'

        elif lang == "Ja":
            if not os.path.exists("trans_result/ja"):
                os.makedirs("trans_result/ja")

            if not os.path.exists("trans_result/ja/v5"):
                os.makedirs("trans_result/ja/v5")
            self.question_file_path = \
                "../data/ja_question_v5.csv"
            self.source_context_file_path = \
                "../data/ja_question_v5_context.csv"
            trans_result_dir_name = \
                os.path.join("trans_result/ja/v5", embedding_name)

        if self.beam_search == True:
            trans_result_dir_name = os.path.join(trans_result_dir_name, "beam")

        if not os.path.exists(trans_result_dir_name):
            os.makedirs(trans_result_dir_name)
        self.question_trans_file_path = os.path.join(
            trans_result_dir_name, "TRANS.question.txt.new")
        self.trans_context_file_path = os.path.join(
            trans_result_dir_name, "TRANS.txt.new")
        self.context_attention_file_path = os.path.join(
            trans_result_dir_name, "ATTN.txt.new")

        self.lang = lang

    def get_japanese_answers_with_attention(self, id2answerindices_dict):
        ja_ans = {}
        for k, v in id2answerindices_dict.items():
            start, end = v
            title = self.question_id2title[k]["title"]
            para_idx = self.question_id2title[k]["para_idx"]
            attention_weights_list = self.title2attention_dic[title][para_idx]

            ans = []
            ans_idx = []

            paragraph = {}
            paragraph_sent_length = []

            for i, sent_idx in enumerate(self.sent_idx_dic[title][para_idx]):
                paragraph[i] = self.trans_lines[sent_idx]
                paragraph_sent_length.append(
                    len(self.trans_lines[sent_idx].split()))
            for i in range(int(start), int(end) + 1):
                m, n = 0, 0
                prev_idx = 0
                for j in range(len(paragraph_sent_length)):
                    if prev_idx + (paragraph_sent_length[j] - 1) < i:
                        prev_idx += paragraph_sent_length[j]
                    else:
                        m = j
                        n = i - prev_idx
                        break
                sentence_idx = self.sent_idx_dic[title][para_idx][m]
                # TODO: Fix code not to use tokenizer here for multilingual
                # adatation.
                if self.lang == "Fr":
                    source_tokens = self.source_lines[sentence_idx].split()
                elif self.lang == "Ja":
                    source_tokens = tokenize_preprocess_japanese_sent(
                        self.source_lines[sentence_idx]).split(" ")
                attention_weight_vector = attention_weights_list[m][n]
                source_idx = np.argmax(attention_weight_vector)
                if source_idx == len(attention_weight_vector) - 1 and \
                        n != len(self.trans_lines[m]) - 1:
                    source_idx = np.argsort(attention_weight_vector)[::-1][1]

                ans_token = source_tokens[source_idx].replace("\n", "")
                ans_idx.append(source_idx)

            if len(ans_idx) == 0:
                ja_ans[k] = ""
            else:
                start = min(ans_idx)
                end = max(ans_idx)
                ja_ans[k] = " ".join(source_tokens[start:end + 1])

        return ja_ans

    def get_japanese_answers_with_attention_use_soft_alignment(self, id2answerindices_dict, alpha=0.3):
        ja_ans = {}

        for k, v in id2answerindices_dict.items():
            start, end = v
            title = self.question_id2title[k]["title"]
            para_idx = self.question_id2title[k]["para_idx"]
            attention_weights_list = self.title2attention_dic[title][para_idx]

            ans = []
            ans_idx = []

            paragraph = {}
            paragraph_sent_length = []
            paragraph_source_tokens = []

            non_answer_scores = []
            for i, sent_idx in enumerate(self.sent_idx_dic[title][para_idx]):
                paragraph[i] = self.trans_lines[sent_idx]
                paragraph_sent_length.append(
                    len(self.trans_lines[sent_idx].split()))

                if self.lang == "Fr":
                    source_tokens = self.source_lines[sent_idx].split()
                elif self.lang == "Ja":
                    source_tokens = tokenize_preprocess_japanese_sent(
                        self.source_lines[sent_idx]).split(" ")

                non_answer_scores.append(
                    [1.0 for source_token in source_tokens])
                # [[1.0,1.0, ....]], shape : (num_sent_paragraph x num_source_token)
                paragraph_source_tokens.append(source_tokens)
                # [[私は,, ....]], shape : (num_sent_paragraph x num_source_token)

            for i in range(int(start), int(end) + 1):
                m, n = 0, 0
                prev_idx = 0
                for j in range(len(paragraph_sent_length)):
                    if prev_idx + (paragraph_sent_length[j] - 1) < i:
                        prev_idx += paragraph_sent_length[j]
                    else:
                        m = j
                        n = i - prev_idx
                        break

                for j in range(len(paragraph_source_tokens[m])):
                    if len(attention_weights_list[m][n]) <= j:
                        print(paragraph_source_tokens)
                        continue
                    non_answer_scores[m][j] *= (1 -
                                                attention_weights_list[m][n][j])

            answer_scores = []
            for m in range(len(non_answer_scores)):
                answer_scores.append(
                    [1.0 - non_answer_score for non_answer_score in non_answer_scores[m]])

            ans_indices = {}
            ans_tokens = []

            for m in range(len(non_answer_scores)):
                for j in range(len(non_answer_scores[m])):
                    if answer_scores[m][j] > alpha:
                        ans_indices.setdefault(m, [])
                        ans_indices[m].append(j)

            for sent_index, token_indices in ans_indices.items():
                m_start, m_end = min(token_indices), max(token_indices)
                ans_tokens.extend(
                    paragraph_source_tokens[sent_index][m_start:m_end + 1])
            print(ans_tokens)

            if len(ans_tokens) == 0:
                for i, sent_idx in enumerate(self.sent_idx_dic[title][para_idx]):
                    paragraph[i] = self.trans_lines[sent_idx]
                    paragraph_sent_length.append(
                        len(self.trans_lines[sent_idx].split()))
                for i in range(int(start), int(end) + 1):
                    m, n = 0, 0
                    prev_idx = 0
                    for j in range(len(paragraph_sent_length)):
                        if prev_idx + (paragraph_sent_length[j] - 1) < i:
                            prev_idx += paragraph_sent_length[j]
                        else:
                            m = j
                            n = i - prev_idx
                            break
                    sentence_idx = self.sent_idx_dic[title][para_idx][m]
                    # TODO: Fix code not to use tokenizer here for multilingual
                    # adatation.
                    if self.lang == "Fr":
                        source_tokens = self.source_lines[sentence_idx].split()
                    elif self.lang == "Ja":
                        source_tokens = tokenize_preprocess_japanese_sent(
                            self.source_lines[sentence_idx]).split(" ")

                    attention_weight_vector = attention_weights_list[m][n]
                    source_idx = np.argmax(attention_weight_vector)
                    if source_idx == len(attention_weight_vector) - 1 and \
                            n != len(self.trans_lines[m]) - 1:
                        source_idx = np.argsort(
                            attention_weight_vector)[::-1][1]

                    ans_token = source_tokens[source_idx].replace("\n", "")
                    ans_idx.append(source_idx)
                    start = min(ans_idx)
                    end = max(ans_idx)
                    ans_tokens = source_tokens[start:end + 1]

            ja_ans[k] = " ".join(ans_tokens)

        return ja_ans

    def get_japanese_answers_with_attention_beam_index(self, id2answerindices_dict):
        ja_ans = {}
        for k, v in id2answerindices_dict.items():
            start, end = v
            title = self.question_id2title[k]["title"]
            para_idx = self.question_id2title[k]["para_idx"]

            ans = []
            ans_idx = []

            paragraph = {}
            paragraph_sent_length = []

            for i, sent_idx in enumerate(self.sent_idx_dic[title][para_idx]):
                paragraph[i] = self.trans_lines[sent_idx]
                paragraph_sent_length.append(
                    len(self.trans_lines[sent_idx].split()))

            for i in range(int(start), int(end) + 1):
                m, n = 0, 0
                prev_idx = 0
                for j in range(len(paragraph_sent_length)):
                    if prev_idx + (paragraph_sent_length[j] - 1) < i:
                        prev_idx += paragraph_sent_length[j]
                    else:
                        m = j
                        n = i - prev_idx
                        break
                sentence_idx = self.sent_idx_dic[title][para_idx][m]
                # TODO: Fix code not to use tokenizer here for multilingual
                # adatation.
                if self.lang == "Fr":
                    source_tokens = self.source_lines[sentence_idx].split()
                elif self.lang == "Ja":
                    source_tokens = tokenize_preprocess_japanese_sent(
                        self.source_lines[sentence_idx]).split(" ")

                source_idx = self.attention_matrix[sentence_idx][n]

                ans_token = source_tokens[source_idx].replace("\n", "")
                ans_idx.append(source_idx)

            if len(ans_idx) == 0:
                print("NO ANSWER")
                print(k)
                ja_ans[k] = ""
            else:
                start = min(ans_idx)
                end = max(ans_idx)
                ja_ans[k] = " ".join(source_tokens[start:end + 1])

        return ja_ans


class SQuADMLAnswerRetreivalBing(object):
    def __init__(self, lang, version=3, embedding_name='bing'):
        self.set_question_context_answer_file(lang, version, embedding_name)
        
        self.sent_idx_dic = create_sent_idx_dic(
            self.source_context_file_path)
        
        self.question_id2title = create_questionid2title_dic(
            self.question_file_path)
        
        self.source_lines, self.trans_lines = \
            get_source_lines_trans_lines(
                self.source_context_file_path, self.trans_context_file_path)

        if self.lang == "Fr" or self.lang == "De":
            self.source_lines = open(
                self.japanese_context_file_path, "r").readlines()
            self.source_lines = [sentence.rstrip()
                                 for sentence in self.source_lines]

    def get_id2answerindices_dict(self, answer_spans, question_ids):
        id2answerindices_dict = {int(id_): span
                                 for id_, span in
                                 zip(question_ids, answer_spans)}
        return id2answerindices_dict

    def load_alignment_data(self, alignment_json_fp):
        """
        each line containts the word alignment information from Microsoft Bing.
        e.g., '0:2-14:19 4:6-21:25 8:8-0:3 9:9-5:8 11:13-27:30 14:14-10:12'
        1. Parse the information to each alignment pair.
            ['0:2-14:19', '4:6-21:25', '8:8-0:3', '9:9-5:8', '11:13-27:30', '14:14-10:12']
        2. Store it into dictionary.
            {(14, 19): (0, 2),
            (21, 25): (4, 6),
            (0, 3): (8, 8),
            (5, 8): (9, 9),
            (27, 30): (11, 13),
            (10, 12): (14, 14)}
        3. Store it into another dictionary. m is sentence index.
            { m : {(14, 19): (0, 2),
            (21, 25): (4, 6),
            (0, 3): (8, 8),
            (5, 8): (9, 9),
            (27, 30): (11, 13),
            (10, 12): (14, 14)}
        """
        print("load alignment_data has been callsed.")
        alignment_lines = open(alignment_json_fp, "r").readlines()
        sent_index_to_align = {}
        for i, line in enumerate(alignment_lines):
            align_info = line.split()

            align_dict = {}
            for item in align_info:
                source, target = item.split("-")
                source = source.split(":")
                target = target.split(":")
            
                align_dict[(int(target[0]), int(target[1]))] = \
                    (int(source[0]), int(source[1]))
                sent_index_to_align[i] = align_dict
        
        print(sent_index_to_align)
        return sent_index_to_align

    def set_question_context_answer_file(self, lang, version, embedding_name):
        if lang == "Fr":
            if not os.path.exists("trans_result/fr/v5"):
                os.makedirs("trans_result/fr/v5")
            self.question_file_path = \
                "../data/fr_question_v5.csv"
            self.source_context_file_path = \
                "../data/fr_question_v5_context.csv"
            trans_result_dir_name = \
                os.path.join("trans_result/fr/v5", embedding_name)

            # For french, you need to you pre-tokenized file.
            self.japanese_context_file_path = '../nmt/french_context_tmp.txt'

        elif lang == "Ja":
            if not os.path.exists("trans_result/ja"):
                os.makedirs("trans_result/ja")

            if not os.path.exists("trans_result/ja/v5"):
                os.makedirs("trans_result/ja/v5")
            self.question_file_path = \
                "../data/ja_question_v5.csv"
            self.source_context_file_path = \
                "../data/ja_question_v5_context.csv"
            trans_result_dir_name = \
                os.path.join("trans_result/ja/v5", embedding_name)

        if not os.path.exists(trans_result_dir_name):
            os.makedirs(trans_result_dir_name)
        self.question_trans_file_path = os.path.join(
            trans_result_dir_name, "TRANS.question.txt.new")
        self.trans_context_file_path = os.path.join(
            trans_result_dir_name, "TRANS.txt.new")
        self.context_attention_file_path = os.path.join(
            trans_result_dir_name, "ATTN.txt.new")

        self.alignment = self.load_alignment_data(self.context_attention_file_path)

        self.lang = lang

    def get_japanese_answers_with_attention(self, id2answerindices_dict, id2answer_dict):
        ja_ans = {}
        for k, v in id2answerindices_dict.items():
            start, end = v
            title = self.question_id2title[k]["title"]
            para_idx = self.question_id2title[k]["para_idx"]

            ans = []
            ans_idx = []

            paragraph = {}
            paragraph_sent_length = []


            for i, sent_idx in enumerate(self.sent_idx_dic[title][para_idx]):
                paragraph[i] = self.trans_lines[sent_idx]
                paragraph_sent_length.append(
                    len(self.trans_lines[sent_idx].split()))
            
            for i in range(int(start), int(end) + 1):
                m, n = 0, 0
                prev_idx = 0
                for j in range(len(paragraph_sent_length)):
                    if prev_idx + (paragraph_sent_length[j] - 1) < i:
                        prev_idx += paragraph_sent_length[j]
                    else:
                        m = j
                        n = i - prev_idx
                        break
                sentence_idx = self.sent_idx_dic[title][para_idx][m]
                index_in_translated_gt = [n, n + len(id2answer_dict[k])]
                break

            source_sent = self.source_lines[sentence_idx]
            # Get alignment information
            align_info = self.alignment[sentence_idx]
            # Get the indices in the English Paragraph
            predicted_answer = id2answer_dict[k]

            translated_sent = self.trans_lines[sentence_idx]
            index_in_translated = []
            for i in range(index_in_translated_gt[0], index_in_translated_gt[1]):
                for index in align_info.keys():
                    if i == index[0]:
                        index_in_translated.append(index)
            
            index_in_source = []
            for index in index_in_translated:
                if index in align_info.keys():
                    index_in_source.append(align_info[index])

            if len(index_in_source) == 0:
                ja_ans[k] = id2answer_dict[k]
            else:
                start = min([index[0] for index in index_in_source])
                end = max([index[1] for index in index_in_source])

            ja_ans[k] = source_sent[start:end + 1]

        return ja_ans
