import os
import argparse
import numpy as np
import json
import re
from collections import defaultdict

import string

def make_vocab_answers(input_dir, n_answers):
    """Make dictionary for top n answers and save them into text file."""
    answers = defaultdict(lambda: 0)
    datasets = ['train_balanced_questions.statement.jsonl','testdev_balanced_questions.statement.jsonl','val_balanced_questions.statement.jsonl']
    for dataset in datasets:

        with open(input_dir + '/' + dataset, "r", encoding="utf-8") as fin:
            print("start ",dataset)
            for line in fin:
                annotations = json.loads(line)
                # try:
                word = annotations['question']['choices']['text']
                if re.search(r"[^\w\s]", word):
                    continue
                answers[word] += 1
                # except:
                #     print(annotations['question']['choices']['text'])
                #     break


    answers =  sorted(answers,key=lambda x:answers[x], reverse=True)

    assert ('<unk>' not in answers)
    top_answers = ['<unk>'] + answers[:n_answers - 1]  # '-1' is due to '<unk>'

    with open('/data2/KJE/GQA/vocab_answers.txt', 'w') as f:
        f.writelines([w + '\n' for w in top_answers])

    print('Make vocabulary for answers')
    print('The number of total words of answers: %d' % len(answers))
    print('Keep top %d answers into vocab' % n_answers)




def main(args):
    input_dir = '/data2/KJE/GQA'
    n_answers = 10000
    make_vocab_answers(input_dir + '/statement', n_answers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/run/media/hoosiki/WareHouse3/mtb/datasets/VQA',
                        help='directory for input questions and answers')
    parser.add_argument('--n_answers', type=int, default=1000,
                        help='the number of answers to be kept in vocab')
    args = parser.parse_args()
    main(args)