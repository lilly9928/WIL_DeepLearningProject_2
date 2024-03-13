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

    # # 관계를 포함하여 출력
    # min_length = min(len(relations), len(top_labels))  # 둘 중 더 짧은 길이를 기준으로 함
    # result = [(relations[i][0], relations[i][1], top_labels[i]) for i in range(min_length)]
    return top_labels



def concepts_to_adj_matrices_2hop_all_pair__use_LM__Part2(data):
    qc_ids, ac_ids, question, extra_nodes = data
    cid2score = get_LM_score(qc_ids+ac_ids+extra_nodes, question)

    return (qc_ids, ac_ids, question, extra_nodes, cid2score)


# def fetch_conceptnet_graph(token):
#     url = f"http://api.conceptnet.io/c/en/{token}"
#     response = requests.get(url)
#     data = response.json()
#     edges = []
#     for edge in data['edges']:
#         start = edge['start']['label']
#         end = edge['end']['label']
#         weight = edge['weight']
#         edges.append([start, end, weight])
#     edges.sort(key=lambda x: x[2], reverse=True)
#     edges = edges[:10]
#
#     return edges


def fetch_conceptnet_graph(word, limit=3):
    url = f'http://api.conceptnet.io/c/en/{word}?limit={limit}'
    response = requests.get(url)
    edges = []

    if response.status_code == 200:
        data = response.json()
        for edge in data['edges']:
            edges.append({
                'start': edge['start']['label'],
                'end': edge['end']['label'],
                'rel': edge['rel']['label'],
                'weight': edge['weight']
            })
    return edges
def get_conceptnet_relations(token):
    url = f"http://api.conceptnet.io/query?node=/c/en/{token}"
    response = requests.get(url)
    data = response.json()
    edges = []
    for edge in data['edges']:
        if bool(re.match('^[a-zA-Z]+$', edge['start']['label'])) and  bool(re.match('^[a-zA-Z]+$', edge['end']['label'])):
            subject = edge['end']['label']
            relation = edge['rel']['label']
            object = edge['start']['label']
            weight = edge['weight']


            edges.append([subject,relation,object,weight])
    edges.sort(key=lambda x: x[3], reverse=True)
    edges = edges[:10]

    return edges


import re

def convert_str_list_to_dict(str_list):
    # 주어진 형식에 맞는 패턴 정의
    pattern = r'\[\[(.*?)\]\]'

    # STR list에서 subject와 object 추출

    subject, object_ = re.findall(pattern, str_list)

# relation 추출
    relation = str_list.split(']]')[1].split('[[')[0]

# 딕셔너리 생성
    result = {'subject': subject, 'relation': relation, 'object': object_}

    return result



def extract_pos_tokens(sentence):
    # tokens = re.findall(r'\b(?!(?:is|am|are|was|were|be|been|being)\b)\b(?!(?:\w+ly)\b)(\w+)\b', sentence)
    tokens = word_tokenize(sentence)
    tagged_tokens = pos_tag(tokens)
    tokens = [word for word, pos in tagged_tokens if pos in ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"]]

    print(tokens)

    return tokens

def edges2node(tokens):
    node = []
    for token in tokens:
        edges = fetch_conceptnet_graph(token)
        if len(edges) > 0:
            for edge in edges:
                edge_remove_weight = edge[:-1]
                node.append(edge_remove_weight[0])
                node.append(edge_remove_weight[1])

    return node


def get_conceptnet_data(word):
    """
    ConceptNet API를 사용하여 주어진 단어에 대한 관련 데이터를 가져옵니다.
    """
    # ConceptNet API 엔드포인트 및 쿼리 구성
    endpoint = f"http://api.conceptnet.io/c/en/{word}?offset=0&limit=1000"

    # API 요청을 보내고 응답을 가져옴
    response = requests.get(endpoint)

    # 응답이 성공인 경우 데이터를 JSON 형식으로 반환
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("ConceptNet API에 대한 요청이 실패하였습니다.")
        return None

def concepts_to_adj_matrices_2hop_all_pair__use_LM__Part1(data):
    qc_ids, ac_ids, question = data

    # 질문 답변 노드
    qa_nodes = set(qc_ids) | set(ac_ids)

    # 추가 노드 집합
    extra_nodes = set()

    # 이미 처리한 노드들의 ConceptNet 데이터를 저장할 딕셔너리 초기화
    conceptnet_cache = {}
    q_data_relation=[]
    a_data_relation = []

    # 모든 질문과 답변 노드들에 대해 2-hop 이웃
    for qid in qa_nodes:
        for aid in qa_nodes:
            if qid != aid:
                if qid in conceptnet_cache:
                    q_data = conceptnet_cache[qid]

                else:
                    q_data = get_conceptnet_relations(qid)
                    conceptnet_cache[qid] = q_data

                if aid in conceptnet_cache:
                    a_data = conceptnet_cache[aid]

                else:
                    a_data = get_conceptnet_relations(aid)
                    conceptnet_cache[aid] = a_data


                if q_data and a_data:

                    for relation in q_data:
                        if relation[0] == qid:
                            q_related = relation
                            extra_nodes.add(q_related[2])

                    for relation in q_data:
                        if relation[0] == aid:
                            a_related = relation
                            extra_nodes.add(a_related[2])

    return (sorted(qc_ids), sorted(ac_ids), question, sorted(extra_nodes))


# 데이터 처리 함수
def process_data(item):
    #data 가져오기
    question = item['question']['stem']
    caption = item['image_caption']
    object_class = item["object_class"]

    #토큰 추출
    q_tokens = extract_pos_tokens(question)
    i_tokens = extract_pos_tokens(caption)

    #질문, 캡션 토큰 합치기
    tokens = list(set(q_tokens+i_tokens))
    data = (tokens, object_class, question)

    result = concepts_to_adj_matrices_2hop_all_pair__use_LM__Part1(data)
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
    import requests
    import networkx as nx
    import matplotlib.pyplot as plt



    # 주어진 단어 리스트
    words = ['red', 'picture', 'apple', 'tree']

    # 그래프 생성
    G = nx.Graph()

    # 각 단어에 대해 ConceptNet에서 정보 조회 및 그래프에 노드와 엣지 추가
    for word in words:
        G.add_node(word)  # 주어진 단어를 노드로 추가
        edges = fetch_conceptnet_graph(word)
        for edge in edges:
            # 관련 있는 단어도 노드로 추가
            G.add_node(edge['end'])
            # 주어진 단어와 관련 있는 단어를 엣지로 연결
            G.add_edge(word, edge['end'], label=edge['rel'])

    # 노드 위치 결정
    pos = nx.spring_layout(G)

    # 그래프 그리기
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'))
    plt.title("Graph Visualization with ConceptNet Relations", size=15)
    plt.show()

    # import spacy
    # import requests
    #
    # # spaCy 모델 로드
    # nlp = spacy.load("en_core_web_sm")
    #
    # # 데이터
    # question = "Is it overcast?"
    # sentence = "The image is a photography of a snowboarder in mid air."
    # objects = ["snowboard", "person"]
    #
    #
    # # 핵심 단어 추출 함수
    # def extract_key_words(text):
    #     doc = nlp(text)
    #     # 명사와 형용사를 주요 단어로 간주
    #     key_words = [token.text for token in doc if token.pos_ in ['NOUN', 'ADJ']]
    #     return key_words
    #
    #
    # # ConceptNet에서 관련 정보를 조회하는 함수
    # def fetch_conceptnet_edges(word, limit=5):
    #     url = f'http://api.conceptnet.io/c/en/{word}?limit={limit}'
    #     response = requests.get(url)
    #     edges = []
    #
    #     if response.status_code == 200:
    #         data = response.json()
    #         for edge in data['edges']:
    #             edges.append({
    #                 'start': edge['start']['label'],
    #                 'end': edge['end']['label'],
    #                 'rel': edge['rel']['label'],
    #                 'weight': edge['weight']
    #             })
    #     return edges
    #
    #
    # # 질문, 문장, 객체에서 핵심 단어 추출
    # question_key_words = extract_key_words(question)
    # sentence_key_words = extract_key_words(sentence)
    # object_key_words = objects  # 객체 리스트는 이미 주요 단어로 간주
    #
    # # 모든 핵심 단어를 하나의 집합으로 결합하여 중복 제거
    # all_key_words = set(question_key_words + sentence_key_words + object_key_words)
    #
    # # 각 핵심 단어에 대해 ConceptNet에서 정보 조회
    # for word in all_key_words:
    #     print(f"Word: {word}")
    #     edges = fetch_conceptnet_edges(word)
    #     for edge in edges:
    #         print(f"  - {edge['start']} {edge['rel']} {edge['end']} (weight: {edge['weight']})")
    #     print("\n")

    #
    # import spacy
    # import requests
    #
    # # spaCy 모델 로드
    # nlp = spacy.load("en_core_web_sm")
    #
    #
    # # ConceptNet에서 관련 노드를 조회하는 함수
    # def fetch_conceptnet_edges(word, limit=5):
    #     url = f'http://api.conceptnet.io/c/en/{word}?limit={limit}'
    #     response = requests.get(url)
    #     edges = []
    #
    #     if response.status_code == 200:
    #         data = response.json()
    #         for edge in data['edges']:
    #             start = edge['start']['label']
    #             end = edge['end']['label']
    #             rel = edge['rel']['label']
    #             weight = edge['weight']
    #             edges.append((start, rel, end, weight))
    #     return edges
    #
    #
    # # 분석할 문장들
    # sentences = [
    #     "Is it overcast?",
    #     "the image is a photography of a snowboarder in mid air",
    #     "snowboard,person"
    # ]
    #
    #
    # # 명사와 형용사를 추출하는 함수
    # def extract_specific_words(sentence, target_pos):
    #     doc = nlp(sentence)
    #     words = [token.text for token in doc if token.pos_ in target_pos]
    #     return words
    #
    #
    # # 필터링할 품사 설정 (명사와 형용사)
    # target_pos = ['NOUN', 'ADJ']
    # unique_words = set()
    #
    # # 각 문장에서 특정 단어 추출 및 중복 제거
    # for sentence in sentences:
    #     unique_words.update(extract_specific_words(sentence, target_pos))
    #
    # # 추출된 단어들과 관련된 ConceptNet 노드 조회
    # for word in unique_words:
    #     print(f"Word: {word}")
    #     edges = fetch_conceptnet_edges(word)
    #     for start, rel, end, weight in edges:
    #         print(f"  - {start} {rel} {end} (weight: {weight})")
    #     print("\n")



    # import spacy
    # from graphviz import Digraph
    #
    # # spaCy 모델 로드
    # nlp = spacy.load("en_core_web_sm")
    #
    # # 분석할 문장
    # sentences = [
    #     "What is red in this picture?",
    #     "The picture shows a red apple and a green tree."
    # ]
    #
    # for index, sentence in enumerate(sentences, start=1):
    #     doc = nlp(sentence)
    #     dot = Digraph(comment='Dependency Tree')
    #
    #     # 그래프 스타일 설정
    #     dot.attr('node', shape='ellipse', style='filled', color='lightblue', fontname='Helvetica')
    #     dot.attr('edge', fontname='Helvetica')
    #
    #     for token in doc:
    #         # 현재 토큰의 텍스트와 품사를 노드로 추가
    #         node_label = f"{token.text}\n[{token.pos_}]"
    #         dot.node(name=str(token.i), label=node_label)
    #
    #         # 의존성 관계를 화살표로 추가
    #         if token.dep_ != "ROOT":
    #             dot.edge(str(token.head.i), str(token.i), label=token.dep_, fontsize='11', fontcolor='darkgreen')
    #
    #     # 파일 이름 설정
    #     filename = f'dependency_tree_{index}'
    #
    #     # 그래프를 PNG 이미지로 저장하고 시각화
    #     dot.render(filename, format='png', cleanup=True)
    #     print(f"{filename}.png has been created.")





    # data 가져오기
    # question = "What is red in this picture?"
    # caption = "The picture shows a red apple and a green tree."
    # object_class = "apple"

    # 토큰 추출
    # q_tokens = extract_pos_tokens(question)
    # i_tokens = extract_pos_tokens(caption)
    #
    # # 질문, 캡션 토큰 합치기
    # tokens = list(set(q_tokens + i_tokens))
    # data = (tokens, object_class, question)
    #
    # result = concepts_to_adj_matrices_2hop_all_pair__use_LM__Part1(data)
    # *result2, node = concepts_to_adj_matrices_2hop_all_pair__use_LM__Part2(result)
    #
    # print(node)


# if __name__ == '__main__':

# input_path = '/data2/KJE/GQA/statement/testdev_balanced_questions_object.statement.jsonl'
# output_path = '/data2/KJE/GQA/statement/testdev_balanced_questions_object_relation_new.statement.jsonl'
#     multiprocessing.set_start_method('spawn')
#     main(input_path,output_path)

