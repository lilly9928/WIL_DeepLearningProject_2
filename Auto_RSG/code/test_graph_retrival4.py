from graph_retrival4 import extract_key_words, concepts_to_adj_matrices_2hop_all_pair__use_LM__Part1, concepts_to_adj_matrices_2hop_all_pair__use_LM__Part2

# 테스트 케이스
question = "Is it overcast?"
sentence = "a snowboarder jumping in the air in a sunny day"

# 핵심 단어 추출 (가상의 함수 호출로 표현)
question_key_words = extract_key_words(question)  # ['overcast']
sentence_key_words = extract_key_words(sentence)  # ['snowboarder', 'jumping', 'air', 'sunny', 'day']

# Part1 함수 호출
part1_result = concepts_to_adj_matrices_2hop_all_pair__use_LM__Part1(question_key_words, sentence_key_words, question)

# Part2 함수 호출
part2_result = concepts_to_adj_matrices_2hop_all_pair__use_LM__Part2(part1_result)

# 결과 출력 (디버깅 목적)
print(part2_result)
