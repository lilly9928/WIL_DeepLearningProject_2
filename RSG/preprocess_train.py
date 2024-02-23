import os
import argparse
from multiprocessing import cpu_count
from utils.convert_csqa import convert_to_entailment
from utils.convert_obqa import convert_to_obqa_statement
from utils.conceptnet import extract_english, construct_graph
from utils.grounding import create_matcher_patterns, ground
from utils.graph import generate_adj_data_from_grounded_concepts__use_LM

os.environ["CUDA_VISIBLE_DEVICES"]= '3'

input_paths = {
       'vcr': {
        'train0': '/data2/KJE/VCR/statement/train0_1.statement.jsonl',
        'train1': '/data2/KJE/VCR/statement/train1_1.statement.jsonl',
        'train2': '/data2/KJE/VCR/statement/train2_1.statement.jsonl',
        'train3': '/data2/KJE/VCR/statement/train3_1.statement.jsonl',
        'train4': '/data2/KJE/VCR/statement/train4_1.statement.jsonl',
        'dev': '/data2/KJE/VCR/statement/val.statement.jsonl',
        'test': '/data2/KJE/VCR/statement/test.statement.jsonl',
        'sample': '/data2/KJE/VCR/statement/sample.statement.jsonl',

        'testdev_balanced_questions': '/data2/KJE/GQA/statement/testdev_balanced_questions.statement.json',
        'train_balanced_questions': '/data2/KJE/GQA/statement/train_balanced_questions.statement.json',
        'val_balanced_questions': '/data2/KJE/GQA/statement/val_balanced_questions.statement.json',

    },
       



    'cpnet': {
        'csv': '/data2/KJE/RSG/data/cpnet/conceptnet-assertions-5.6.0.csv',
    },
}

output_paths = {
    'cpnet': {
        'csv': '/data2/KJE/RSG/data/cpnet/conceptnet.en.csv',
        'vocab': '/data2/KJE/RSG/data/cpnet/concept.txt',
        'patterns': '/data2/KJE/RSG/data/cpnet/matcher_patterns.json',
        'unpruned-graph': '/data2/KJE/RSG/data/cpnet/conceptnet.en.unpruned.graph',
        'pruned-graph': '/data2/KJE/RSG/data/cpnet/conceptnet.en.pruned.graph',
    },
     'vcr': {
        'statement': {
            'train0': '/data2/KJE/VCR/statement/train0_1.statement.jsonl',
            'train1': '/data2/KJE/VCR/statement/train1_1.statement.jsonl',
            'train2': '/data2/KJE/VCR/statement/train_2.statement.jsonl',
            'train3': '/data2/KJE/VCR/statement/train_3.statement.jsonl',
            'train4': '/data2/KJE/VCR/statement/train_4.statement.jsonl',

            'train_all': '/data2/KJE/VCR/statement/train_all.statement.jsonl',


            'dev': '/data2/KJE/VCR/statement/val.statement.jsonl',
            'test': '/data2/KJE/VCR/statement/test.statement.jsonl',
            'sample': '/data2/KJE/VCR/statement/sample.statement.jsonl',

            'testdev_balanced_questions': '/data2/KJE/GQA/statement/testdev_balanced_questions.statement.json',
            'train_balanced_questions': '/data2/KJE/GQA/statement/train_balanced_questions.statement.json',
            'val_balanced_questions': '/data2/KJE/GQA/statement/val_balanced_questions.statement.json',

        },
        'grounded': {
            'train0': '/data2/KJE/VCR/grounded/train0_1.grounded.jsonl',
            'train1': '/data2/KJE/VCR/grounded/train1_1.grounded.jsonl',
            'train2': '/data2/KJE/VCR/grounded/train2.grounded.jsonl',
            'train3': '/data2/KJE/VCR/grounded/train3.grounded.jsonl',
            'train4': '/data2/KJE/VCR/grounded/train4.grounded.jsonl',

            'train_all': '/data2/KJE/VCR/grounded/train_all.grounded.jsonl',

            'dev': '/data2/KJE/VCR/grounded/dev.grounded.jsonl',
            'test': '/data2/KJE/VCR/grounded/test.grounded.jsonl',
            'sample': '/data2/KJE/VCR/grounded/sample.grounded.jsonl',

            'testdev_balanced_questions': '/data2/KJE/GQA/grounded/testdev_balanced_questions.grounded.json',
            'train_balanced_questions': '/data2/KJE/GQA/grounded/train_balanced_questions.grounded.json',
            'val_balanced_questions': '/data2/KJE/GQA/grounded/val_balanced_questions.grounded.json',
            
        },
        'graph': {
            'adj-train0': '/data2/KJE/VCR/graph/train0_1.graph.adj.pk',
            'adj-train1': '/data2/KJE/VCR/graph/train1_1.graph.adj.pk',
            'adj-train2': '/data2/KJE/VCR/graph/train2.graph.adj.pk',
            'adj-train3': '/data2/KJE/VCR/graph/train3.graph.adj.pk',
            'adj-train4': '/data2/KJE/VCR/graph/train4.graph.adj.pk',

            'adj-train_all': '/data2/KJE/VCR/graph/train_all.graph.adj.pk',

            'adj-dev': '/data2/KJE/VCR/graph/dev.graph.adj.pk',
            'adj-test': '/data2/KJE/VCR/graph/test.graph.adj.pk',
            'adj-sample': '/data2/KJE/VCR/graph/sample.graph.adj.pk',

            'adj-testdev_balanced_questions': '/data2/KJE/GQA/graph/testdev_balanced_questions.graph.adj.pk',
            'adj-train_balanced_questions': '/data2/KJE/GQA/graph/train_balanced_questions.graph.adj.pk',
            'adj-val_balanced_questions': '/data2/KJE/GQA/graph/val_balanced_questions.graph.adj.pk',
            
        },
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['vcr'], choices=['common', 'csqa', 'hswag', 'anli', 'exp', 'scitail', 'phys', 'socialiqa', 'obqa', 'obqa-fact', 'make_word_vocab'], nargs='+')
    parser.add_argument('--path_prune_threshold', type=float, default=0.12, help='threshold for pruning paths')
    parser.add_argument('--max_node_num', type=int, default=200, help='maximum number of nodes per graph')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
            'vcr': [

            {'func': ground, 'args': (output_paths['vcr']['statement']['train_all'], output_paths['cpnet']['vocab'],
                                          output_paths['cpnet']['patterns'], output_paths['vcr']['grounded']['train_all'],
                                          args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['vcr']['grounded']['train_all'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vcr']['graph']['adj-train_all'], args.nprocs)},
    
        ]
       
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()
    # pass
