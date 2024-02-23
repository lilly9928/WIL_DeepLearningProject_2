import os
import argparse
from multiprocessing import cpu_count
from utils.convert_csqa import convert_to_entailment
from utils.convert_obqa import convert_to_obqa_statement
from utils.conceptnet import extract_english, construct_graph
from utils.grounding import create_matcher_patterns, ground
from utils.graph import generate_adj_data_from_grounded_concepts__use_LM



input_paths = {
    'csqa': {
        'train': './data/csqa/train_rand_split.jsonl',
        'dev': './data/csqa/dev_rand_split.jsonl',
        'test': './data/csqa/test_rand_split_no_answers.jsonl',
    },
       'vcr': {
        'train': '/data2/KJE/VCR/statement/train.statement.jsonl',
        'dev': '/data2/KJE/VCR/statement/dev.statement.jsonl',
        'test': '/data2/KJE/VCR/statement/test.statement.jsonl',
        'sample': '/data2/KJE/VCR/statement/sample.statement.jsonl',
    },
    'obqa': {
        'train': './data/obqa/OpenBookQA-V1-Sep2018/Data/Main/train.jsonl',
        'dev': './data/obqa/OpenBookQA-V1-Sep2018/Data/Main/dev.jsonl',
        'test': './data/obqa/OpenBookQA-V1-Sep2018/Data/Main/test.jsonl',
    },
    'obqa-fact': {
        'train': './data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/train_complete.jsonl',
        'dev': './data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/dev_complete.jsonl',
        'test': './data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/test_complete.jsonl',
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
    'csqa': {
        'statement': {
            'train': './data/csqa/statement/train.statement.jsonl',
            'dev': './data/csqa/statement/dev.statement.jsonl',
            'test': './data/csqa/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': './data/csqa/grounded/train.grounded.jsonl',
            'dev': './data/csqa/grounded/dev.grounded.jsonl',
            'test': './data/csqa/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': './data/csqa/graph/train.graph.adj.pk',
            'adj-dev': './data/csqa/graph/dev.graph.adj.pk',
            'adj-test': './data/csqa/graph/test.graph.adj.pk',
        },
    },
     'vcr': {
        'statement': {
            'train': '/data2/KJE/VCR/statement/train.statement.jsonl',
            'dev': '/data2/KJE/VCR/statement/dev.statement.jsonl',
            'test': '/data2/KJE/VCR/statement/test.statement.jsonl',
            'sample': '/data2/KJE/VCR/statement/sample.statement.jsonl',
        },
        'grounded': {
            'train': '/data2/KJE/VCR/grounded/train.grounded.jsonl',
            'dev': '/data2/KJE/VCR/grounded/dev.grounded.jsonl',
            'test': '/data2/KJE/VCR/grounded/test.grounded.jsonl',
            'sample': '/data2/KJE/VCR/grounded/sample.grounded.jsonl',
        },
        'graph': {
            'adj-train': '/data2/KJE/VCR/graph/train.graph.adj.pk',
            'adj-dev': '/data2/KJE/VCR/graph/dev.graph.adj.pk',
            'adj-test': '/data2/KJE/VCR/graph/test.graph.adj.pk',
            'adj-sample': '/data2/KJE/VCR/graph/sample.graph.adj.pk',
        },
    },
    'obqa': {
        'statement': {
            'train': './data/obqa/statement/train.statement.jsonl',
            'dev': './data/obqa/statement/dev.statement.jsonl',
            'test': './data/obqa/statement/test.statement.jsonl',
            'train-fairseq': './data/obqa/fairseq/official/train.jsonl',
            'dev-fairseq': './data/obqa/fairseq/official/valid.jsonl',
            'test-fairseq': './data/obqa/fairseq/official/test.jsonl',
        },
        'grounded': {
            'train': './data/obqa/grounded/train.grounded.jsonl',
            'dev': './data/obqa/grounded/dev.grounded.jsonl',
            'test': './data/obqa/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': './data/obqa/graph/train.graph.adj.pk',
            'adj-dev': './data/obqa/graph/dev.graph.adj.pk',
            'adj-test': './data/obqa/graph/test.graph.adj.pk',
        },
    },
    'obqa-fact': {
        'statement': {
            'train': './data/obqa/statement/train-fact.statement.jsonl',
            'dev': './data/obqa/statement/dev-fact.statement.jsonl',
            'test': './data/obqa/statement/test-fact.statement.jsonl',
            'train-fairseq': './data/obqa/fairseq/official/train-fact.jsonl',
            'dev-fairseq': './data/obqa/fairseq/official/valid-fact.jsonl',
            'test-fairseq': './data/obqa/fairseq/official/test-fact.jsonl',
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

                # {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['vcr']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vcr']['graph']['adj-dev'], args.nprocs)},
                
                # {'func': ground, 'args': (output_paths['vcr']['statement']['test'], output_paths['cpnet']['vocab'],
                #                           output_paths['cpnet']['patterns'], output_paths['vcr']['grounded']['test'],
                #                           args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['vcr']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vcr']['graph']['adj-dev'], args.nprocs)},
        ],
        'common': [
            {'func': extract_english, 'args': (input_paths['cpnet']['csv'], output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'])},
            {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['unpruned-graph'], False)},
            {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['pruned-graph'], True)},
            {'func': create_matcher_patterns, 'args': (output_paths['cpnet']['vocab'], output_paths['cpnet']['patterns'])},
        ],
       
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()
    # pass
