import requests
import networkx as nx
import matplotlib.pyplot as plt
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import requests
import re
import networkx as nx
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from collections import OrderedDict
import json
from tqdm import tqdm
import multiprocessing
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor


from transformers import RobertaTokenizer, RobertaForMaskedLM

nlp = spacy.load("en_core_web_sm")
class RobertaForMaskedLMwithLoss(RobertaForMaskedLM):
    #
    def __init__(self, config):
        super().__init__(config)
    #
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, masked_lm_labels=None):
        #
        assert attention_mask is not None
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0] #hidden_states of final layer (batch_size, sequence_length, hidden_size)
        prediction_scores = self.lm_head(sequence_output)
        outputs = (prediction_scores, sequence_output) + outputs[2:]
        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            bsize, seqlen = input_ids.size()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)).view(bsize, seqlen)
            masked_lm_loss = (masked_lm_loss * attention_mask).sum(dim=1)
            outputs = (masked_lm_loss,) + outputs
            # (masked_lm_loss), prediction_scores, sequence_output, (hidden_states), (attentions)
        return outputs

print ('loading pre-trained LM...')
TOKENIZER = RobertaTokenizer.from_pretrained('roberta-large')
LM_MODEL = RobertaForMaskedLMwithLoss.from_pretrained('roberta-large')
LM_MODEL.cuda()
LM_MODEL.to('cuda')
# LM_MODEL = torch.nn.DataParallel(LM_MODEL).to('cuda')
LM_MODEL.eval()
print ('loading done')


def fetch_conceptnet_graph(word, limit=100):
    url = f'http://api.conceptnet.io/c/en/{word}?limit={limit}'
    response = requests.get(url)
    edges = []

    if response.status_code == 200:
        data = response.json()
        for edge in data['edges']:
            if re.match(r'^[a-zA-Z ]+$', edge['start']['label']) and re.match(r'^[a-zA-Z ]+$', edge['end']['label']):
                edges.append({
                    'start': edge['start']['label'],
                    'end': edge['end']['label'],
                    'rel': edge['rel']['label'],
                    'weight': edge['weight']
                })
    return edges


def get_edges_for_word(word, limit=100):
    url = f"http://api.conceptnet.io/query?node=/c/en/{word}&limit={limit}"
    response = requests.get(url)
    data = response.json()

    filtered_edges = []

    relationships = [
        "/r/IsA",
        "/r/UsedFor",
        "/r/HasA",
        "/r/AtLocation",
        "/r/HasProperty",
        "/r/LocatedNear",
        "/r/InstanceOf",
        "/r/RelatedTo",
        "/r/MadeOf",
        "/r/PartOf",
        "/r/CapableOf",
        "/r/Causes"
    ]

    for edge in data.get('edges', []):
        rel = edge.get('rel', {}).get('@id', '')
        if rel in relationships:
            start = edge.get('start', {}).get('label', '')
            end = edge.get('end', {}).get('label', '')

            filtered_edges.append({
                'start':start,
                'end': end,
                'rel': rel,
                }
            )

    return filtered_edges


# 핵심 단어 추출 함수
def extract_key_words(text):
    doc = nlp(text)
    # 명사와 형용사를 주요 단어로 간주
    key_words = [token.text for token in doc if token.pos_ in ['NOUN', 'ADJ']]
    return key_words


def graph_generator(all_key_words):
    G = nx.Graph()

    # 각 단어에 대해 ConceptNet에서 정보 조회 및 그래프에 노드와 엣지 추가
    for word in all_key_words:
        G.add_node(word)  # 주어진 단어를 노드로 추가
        edges = fetch_conceptnet_graph(word)
        for edge in edges:
            # 관련 있는 단어도 노드로 추가
            G.add_node(edge['end'])
            
            # 주어진 단어와 관련 있는 단어를 엣지로 연결, 엣지의 레이블과 가중치 추가
            """
            G.add_edge(word, edge['end'], label=edge['rel'], weight=edge['weight'])

            print(f"  - {edge['start']} {edge['rel']} {edge['end']} (weight: {edge['weight']})")
            print("\n")"""
            
            # 주어진 단어와 관련 있는 단어를 엣지로 연결
            G.add_edge(word, edge['end'], label=edge['rel'])

            print(f"  - {edge['start']} {edge['rel']} {edge['end']} (weight: {edge['weight']})")
            print("\n")

    # 노드 위치 결정
    pos = nx.spring_layout(G)

    # 그래프 그리기
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'))
    # 가중치를 엣지 레이블로 표시하고 싶다면, 아래 주석 해제
    # edge_weights = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)
    plt.title("Graph Visualization with ConceptNet Relations", size=15)
    plt.show()




def concepts_to_adj_matrices_2hop_all_pair__use_LM__Part1(question_key_words,sentence_key_words,question):

    # 질문 답변 노드
    qc_nodes = set(question_key_words) | set(sentence_key_words)

    # 추가 노드 집합
    extra_nodes = set()

    # 이미 처리한 노드들의 ConceptNet 데이터를 저장할 딕셔너리 초기화
    conceptnet_cache = {}
    q_data_relation=[]
    a_data_relation = []

    # 모든 질문과 답변 노드들에 대해 2-hop 이웃
    for qid in qc_nodes:
        for sid in qc_nodes:
            if qid != sid:
                if qid in conceptnet_cache:
                    q_data = conceptnet_cache[qid]

                else:
                    q_data = get_edges_for_word(qid)
                    conceptnet_cache[qid] = q_data

                if sid in conceptnet_cache:
                    s_data = conceptnet_cache[sid]

                else:
                    s_data = get_edges_for_word(sid)
                    conceptnet_cache[sid] = s_data


                if q_data and s_data:

                    for relation in q_data:
                        if relation['start'] == qid:
                            q_related = relation
                            extra_nodes.add(q_related['end'])

                    for relation in s_data:
                        if relation['start'] == sid:
                            s_related = relation
                            extra_nodes.add(s_related['end'])

    return (sorted(question_key_words), sorted(sentence_key_words), question, sorted(extra_nodes))


def get_LM_score(cids, question):
    cids = cids[:]
    relations = []  # 관계를 저장할 리스트
    for i in range(len(cids)-1):  # 마지막 노드는 관계를 가져올 필요가 없음
        rel = (cids[i], cids[i + 1])
        relations.append(rel)
    sents, scores = [], []
    for cid in cids:
        if cid == -1:
            sent = question.lower()
        else:
            sent = '{} {}.'.format(question.lower(), cid)
        sent = TOKENIZER.encode(sent, add_special_tokens=True)
        sents.append(sent)
    n_cids = len(cids)
    cur_idx = 0
    batch_size = 50
    while cur_idx < n_cids:
        # Prepare batch
        input_ids = sents[cur_idx: cur_idx + batch_size]
        max_len = max([len(seq) for seq in input_ids])
        for j, seq in enumerate(input_ids):
            seq += [TOKENIZER.pad_token_id] * (max_len - len(seq))
            input_ids[j] = seq
            # Batch 맞추기
        input_ids = torch.tensor(input_ids).cuda()  # [B, seqlen]
        mask = (input_ids != 1).long()  # [B, seq_len]
        # Get LM score
        with torch.no_grad():
            outputs = LM_MODEL(input_ids, attention_mask=mask, masked_lm_labels=input_ids)
            loss = outputs[0]  # [B, ]
            _scores = list(-loss.detach().cpu().numpy())  # List of float
        scores += _scores
        cur_idx += batch_size
    assert len(sents) == len(scores) == len(cids)
    cid2score = OrderedDict(sorted(list(zip(cids, scores)), key=lambda x: -x[1]))  # Score: from high to low

    # 가장 유사도가 높은 노드의 라벨 선택 (상위 10개)
    top_labels = list(cid2score.keys())[:10]  # 가장 유사도가 높은 상위 10개 노드의 라벨

    return top_labels


def concepts_to_adj_matrices_2hop_all_pair__use_LM__Part2(data):
    qc_ids, ac_ids, question, extra_nodes = data
    cid2score = get_LM_score(qc_ids+ac_ids+extra_nodes, question)

    return (qc_ids, ac_ids, question, extra_nodes, cid2score)


def print_n_conceptnet_node(all_key_words):
    for word in all_key_words:
        print(f"Word: {word}")
        edges = fetch_conceptnet_graph(word)
        for edge in edges:
            print(f"  - {edge['start']} {edge['rel']} {edge['end']} (weight: {edge['weight']})")
        print("\n")


def fetch_conceptnet_relations(words, hops=3):
    relations = []
    visited = set()

    for word in words:
        relations.extend(fetch_conceptnet_relations_helper(word, hops, visited))

    return relations

def fetch_conceptnet_relations_helper(word, hops, visited):
    if hops == 0:
        return []

    url = f'http://api.conceptnet.io/query?node=/c/en/{word}'
    response = requests.get(url)

    if response.status_code != 200:
        return []

    data = response.json()
    edges = data['edges']
    relations = []

    for edge in edges:
        start = edge['start']['label']
        end = edge['end']['label']
        rel = edge['rel']['label']

        if start == f'/c/en/{word}' and end not in visited:
            visited.add(end)
            relations.append((word, rel, end))
            relations.extend(fetch_conceptnet_relations_helper(end, hops - 1, visited))

    return relations

def process_data(item):
    #data 가져오기
    question = item['question']['stem']
    caption = item['image_caption']

    question_key_words = extract_key_words(question)
    sentence_key_words = extract_key_words(caption)

    # 모든 핵심 단어를 하나의 집합으로 결합하여 중복 제거
    question_key_words = set(question_key_words)
    sentence_key_words = set(sentence_key_words)

    result = concepts_to_adj_matrices_2hop_all_pair__use_LM__Part1(question_key_words, sentence_key_words, question)
    *result2, node = concepts_to_adj_matrices_2hop_all_pair__use_LM__Part2(result)

    item['relation'] = node

    return item

def main(input_path, output_path):
    input_path = input_path
    output_path = output_path

    # 파일 불러오기
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]

    # 병렬 처리를 위한 프로세스 수 설정
    num_processes = 10

    # 데이터를 프로세스 단위로 분할
    data_chunks = [data[i::num_processes] for i in range(num_processes)]

    # 병렬 처리를 위한 Pool 생성
    pool = multiprocessing.Pool(processes=num_processes)

    updated_data = []

    # 병렬로 데이터 처리 및 결과 수집
    for result in tqdm(pool.imap(process_data, data), total=len(data), desc="Processing data"):
        updated_data.append(result)

    # Pool 종료
    pool.close()
    pool.join()

    # 결과를 파일에 쓰기
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in updated_data:
            # JSONL 형식으로 파일에 쓰기
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

if __name__ == '__main__':
    # 데이터
    # question = "Is it overcast?"
    # sentence = "a snowboarder jumping in the air in a sunny day"

    input_path = '/data2/KJE/GQA/statement/testdev_balanced_questions_statement_promptcap_.jsonl'
    output_path = '/data2/KJE/GQA/statement/testdev_balanced_questions_promptcap_relation_2.statement.jsonl'

    multiprocessing.set_start_method('spawn')
    main(input_path,output_path)
    
    """
    import requests
import networkx as nx
import matplotlib.pyplot as plt
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import requests
import re
import networkx as nx
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from collections import OrderedDict
import json
from tqdm import tqdm
import multiprocessing
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from concurrent.futures import ThreadPoolExecutor


from transformers import RobertaTokenizer, RobertaForMaskedLM

nlp = spacy.load("en_core_web_sm")
class RobertaForMaskedLMwithLoss(RobertaForMaskedLM):
    #
    def __init__(self, config):
        super().__init__(config)
    #
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, masked_lm_labels=None):
        #
        assert attention_mask is not None
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0] #hidden_states of final layer (batch_size, sequence_length, hidden_size)
        prediction_scores = self.lm_head(sequence_output)
        outputs = (prediction_scores, sequence_output) + outputs[2:]
        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            bsize, seqlen = input_ids.size()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)).view(bsize, seqlen)
            masked_lm_loss = (masked_lm_loss * attention_mask).sum(dim=1)
            outputs = (masked_lm_loss,) + outputs
            # (masked_lm_loss), prediction_scores, sequence_output, (hidden_states), (attentions)
        return outputs

print ('loading pre-trained LM...')
TOKENIZER = RobertaTokenizer.from_pretrained('roberta-large')
LM_MODEL = RobertaForMaskedLMwithLoss.from_pretrained('roberta-large')
LM_MODEL.cuda()
LM_MODEL.to('cuda')
# LM_MODEL = torch.nn.DataParallel(LM_MODEL).to('cuda')
LM_MODEL.eval()
print ('loading done')


def fetch_conceptnet_graph(word, limit=100):
    url = f'http://api.conceptnet.io/c/en/{word}?limit={limit}'
    response = requests.get(url)
    edges = []

    if response.status_code == 200:
        data = response.json()
        for edge in data['edges']:
            if re.match(r'^[a-zA-Z ]+$', edge['start']['label']) and re.match(r'^[a-zA-Z ]+$', edge['end']['label']):
                edges.append({
                    'start': edge['start']['label'],
                    'end': edge['end']['label'],
                    'rel': edge['rel']['label'],
                    'weight': edge['weight']
                })
    return edges


def get_edges_for_word(word, limit=100):
    url = f"http://api.conceptnet.io/query?node=/c/en/{word}&limit={limit}"
    response = requests.get(url)
    data = response.json()

    filtered_edges = []

    relationships = [
        "/r/IsA",
        "/r/UsedFor",
        "/r/HasA",
        "/r/AtLocation",
        "/r/HasProperty",
        "/r/LocatedNear",
        "/r/InstanceOf",
        "/r/RelatedTo",
        "/r/MadeOf",
        "/r/PartOf",
        "/r/CapableOf",
        "/r/Causes"
    ]

    for edge in data.get('edges', []):
        rel = edge.get('rel', {}).get('@id', '')
        if rel in relationships:
            start = edge.get('start', {}).get('label', '')
            end = edge.get('end', {}).get('label', '')

            filtered_edges.append({
                'start':start,
                'end': end,
                'rel': rel,
                }
            )

    return filtered_edges


# 핵심 단어 추출 함수
def extract_key_words(text):
    doc = nlp(text)
    # 명사와 형용사를 주요 단어로 간주
    key_words = [token.text for token in doc if token.pos_ in ['NOUN', 'ADJ']]
    return key_words


def graph_generator(all_key_words):
    G = nx.Graph()

    # 각 단어에 대해 ConceptNet에서 정보 조회 및 그래프에 노드와 엣지 추가
    for word in all_key_words:
        G.add_node(word)  # 주어진 단어를 노드로 추가
        edges = fetch_conceptnet_graph(word)
        for edge in edges:
            # 관련 있는 단어도 노드로 추가
            G.add_node(edge['end'])
            # 주어진 단어와 관련 있는 단어를 엣지로 연결
            G.add_edge(word, edge['end'], label=edge['rel'])

            print(f"  - {edge['start']} {edge['rel']} {edge['end']} (weight: {edge['weight']})")
            print("\n")

    # 노드 위치 결정
    pos = nx.spring_layout(G)

    # 그래프 그리기
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'))
    plt.title("Graph Visualization with ConceptNet Relations", size=15)
    plt.show()





def concepts_to_adj_matrices_2hop_all_pair__use_LM__Part1(question_key_words,sentence_key_words,question):

    # 질문 답변 노드
    qc_nodes = set(question_key_words) | set(sentence_key_words)

    # 추가 노드 집합
    extra_nodes = set()

    # 이미 처리한 노드들의 ConceptNet 데이터를 저장할 딕셔너리 초기화
    conceptnet_cache = {}
    q_data_relation=[]
    a_data_relation = []

    # 모든 질문과 답변 노드들에 대해 2-hop 이웃
    for qid in qc_nodes:
        for sid in qc_nodes:
            if qid != sid:
                if qid in conceptnet_cache:
                    q_data = conceptnet_cache[qid]

                else:
                    q_data = get_edges_for_word(qid)
                    conceptnet_cache[qid] = q_data

                if sid in conceptnet_cache:
                    s_data = conceptnet_cache[sid]

                else:
                    s_data = get_edges_for_word(sid)
                    conceptnet_cache[sid] = s_data


                if q_data and s_data:

                    for relation in q_data:
                        if relation['start'] == qid:
                            q_related = relation
                            extra_nodes.add(q_related['end'])

                    for relation in s_data:
                        if relation['start'] == sid:
                            s_related = relation
                            extra_nodes.add(s_related['end'])

    return (sorted(question_key_words), sorted(sentence_key_words), question, sorted(extra_nodes))



def get_LM_score(cids, question):
    cids = cids[:]
    relations = []  # 관계를 저장할 리스트
    for i in range(len(cids)-1):  # 마지막 노드는 관계를 가져올 필요가 없음
        rel = (cids[i], cids[i + 1])
        relations.append(rel)
    sents, scores = [], []
    for cid in cids:
        if cid == -1:
            sent = question.lower()
        else:
            sent = '{} {}.'.format(question.lower(), cid)
        sent = TOKENIZER.encode(sent, add_special_tokens=True)
        sents.append(sent)
    n_cids = len(cids)
    cur_idx = 0
    batch_size = 50
    while cur_idx < n_cids:
        # Prepare batch
        input_ids = sents[cur_idx: cur_idx + batch_size]
        max_len = max([len(seq) for seq in input_ids])
        for j, seq in enumerate(input_ids):
            seq += [TOKENIZER.pad_token_id] * (max_len - len(seq))
            input_ids[j] = seq
            # Batch 맞추기
        input_ids = torch.tensor(input_ids).cuda()  # [B, seqlen]
        mask = (input_ids != 1).long()  # [B, seq_len]
        # Get LM score
        with torch.no_grad():
            outputs = LM_MODEL(input_ids, attention_mask=mask, masked_lm_labels=input_ids)
            loss = outputs[0]  # [B, ]
            _scores = list(-loss.detach().cpu().numpy())  # List of float
        scores += _scores
        cur_idx += batch_size
    assert len(sents) == len(scores) == len(cids)
    cid2score = OrderedDict(sorted(list(zip(cids, scores)), key=lambda x: -x[1]))  # Score: from high to low

    # 가장 유사도가 높은 노드의 라벨 선택 (상위 10개)
    top_labels = list(cid2score.keys())[:10]  # 가장 유사도가 높은 상위 10개 노드의 라벨

    return top_labels



def concepts_to_adj_matrices_2hop_all_pair__use_LM__Part2(data):
    qc_ids, ac_ids, question, extra_nodes = data
    cid2score = get_LM_score(qc_ids+ac_ids+extra_nodes, question)

    return (qc_ids, ac_ids, question, extra_nodes, cid2score)


def print_n_conceptnet_node(all_key_words):
    for word in all_key_words:
        print(f"Word: {word}")
        edges = fetch_conceptnet_graph(word)
        for edge in edges:
            print(f"  - {edge['start']} {edge['rel']} {edge['end']} (weight: {edge['weight']})")
        print("\n")


def fetch_conceptnet_relations(words, hops=3):
    relations = []
    visited = set()

    for word in words:
        relations.extend(fetch_conceptnet_relations_helper(word, hops, visited))

    return relations

def fetch_conceptnet_relations_helper(word, hops, visited):
    if hops == 0:
        return []

    url = f'http://api.conceptnet.io/query?node=/c/en/{word}'
    response = requests.get(url)

    if response.status_code != 200:
        return []

    data = response.json()
    edges = data['edges']
    relations = []

    for edge in edges:
        start = edge['start']['label']
        end = edge['end']['label']
        rel = edge['rel']['label']

        if start == f'/c/en/{word}' and end not in visited:
            visited.add(end)
            relations.append((word, rel, end))
            relations.extend(fetch_conceptnet_relations_helper(end, hops - 1, visited))

    return relations

def process_data(item):
    #data 가져오기
    question = item['question']['stem']
    caption = item['image_caption']

    question_key_words = extract_key_words(question)
    sentence_key_words = extract_key_words(caption)

    # 모든 핵심 단어를 하나의 집합으로 결합하여 중복 제거
    question_key_words = set(question_key_words)
    sentence_key_words = set(sentence_key_words)

    result = concepts_to_adj_matrices_2hop_all_pair__use_LM__Part1(question_key_words, sentence_key_words, question)
    *result2, node = concepts_to_adj_matrices_2hop_all_pair__use_LM__Part2(result)

    item['relation'] = node

    return item

def main(input_path, output_path):
    input_path = input_path
    output_path = output_path

    # 파일 불러오기
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]

    # 병렬 처리를 위한 프로세스 수 설정
    num_processes = 10

    # 데이터를 프로세스 단위로 분할
    data_chunks = [data[i::num_processes] for i in range(num_processes)]

    # 병렬 처리를 위한 Pool 생성
    pool = multiprocessing.Pool(processes=num_processes)

    updated_data = []

    # 병렬로 데이터 처리 및 결과 수집
    for result in tqdm(pool.imap(process_data, data), total=len(data), desc="Processing data"):
        updated_data.append(result)

    # Pool 종료
    pool.close()
    pool.join()

    # 결과를 파일에 쓰기
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in updated_data:
            # JSONL 형식으로 파일에 쓰기
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

if __name__ == '__main__':
    # 데이터
    # question = "Is it overcast?"
    # sentence = "a snowboarder jumping in the air in a sunny day"

    input_path = '/data2/KJE/GQA/statement/testdev_balanced_questions_statement_promptcap_.jsonl'
    output_path = '/data2/KJE/GQA/statement/testdev_balanced_questions_promptcap_relation_2.statement.jsonl'

    multiprocessing.set_start_method('spawn')
    main(input_path,output_path)
    """