import numpy as np
import json
import os
import argparse
import text_helper
from pycocotools.coco import COCO
from collections import defaultdict


def extract_answers(q_answers, valid_answer_set):
    all_answers = [answer["answer"] for answer in q_answers]
    valid_answers = [a for a in all_answers if a in valid_answer_set]
    return all_answers, valid_answers


def vqa_processing(image_dir, annotation_file, annotation_caption_file,question_file, valid_answer_set, image_set):
    print('building vqa %s dataset' % image_set)

    if image_set in ['train2014', 'val2014']:
        annFile = annotation_caption_file%image_set
        coco_caps = COCO(annFile)

        load_answer = True
        with open(annotation_file % image_set) as f:
            annotations = json.load(f)['annotations']
            qid2ann_dict = {ann['question_id']:ann for ann in annotations}

    else:
        load_answer = False

    with open(question_file % image_set) as f:
        questions = json.load(f)['questions']
    coco_set_name = image_set.replace('-dev', '')
    abs_image_dir = os.path.abspath(image_dir % coco_set_name)
    image_name_template = 'COCO_' + coco_set_name + '_%012d'
    dataset = [None] * len(questions)

    unk_ans_count = 0
    for n_q, q in enumerate(questions):
        if (n_q + 1) % 10000 == 0:
            print('processing %d / %d' % (n_q + 1, len(questions)))
        image_id = q['image_id']
        question_id = q['question_id']
        image_name = image_name_template % image_id
        image_path = os.path.join(abs_image_dir, image_name + '.jpg')
        question_str = q['question']
        question_tokens = text_helper.tokenize(question_str)

        iminfo = dict(image_name=image_name,
                      image_path=image_path,
                      question_id=question_id,
                      question_str=question_str,
                      question_tokens=question_tokens)

        if load_answer:
            caption_arr=[]
            caption_arr_tokens=[]
            ann = qid2ann_dict[question_id]
            captionid = coco_caps.getAnnIds(imgIds=image_id)
            for index in range(len(captionid)):
                caption_arr.append(coco_caps.loadAnns(captionid)[index]['caption'])

            for arr in caption_arr:
                caption_arr_tokens.append(text_helper.tokenize(arr))

            all_answers, valid_answers = extract_answers(ann['answers'], valid_answer_set)
            if len(valid_answers) == 0:
                valid_answers = ['<unk>']
                unk_ans_count += 1
            iminfo['all_answers'] = all_answers
            iminfo['valid_answers'] = valid_answers
            iminfo['caption'] = caption_arr
            iminfo['caption_tokens']=caption_arr_tokens

        dataset[n_q] = iminfo
    print('total %d out of %d answers are <unk>' % (unk_ans_count, len(questions)))
    return dataset


def main(args):
    image_dir = 'D:/data/vqa/coco/simple_vqa' + '/Resized_Images/%s/'
    annotation_file = 'D:/data/vqa/coco/simple_vqa' + '/Annotations/v2_mscoco_%s_annotations.json'
    annotation_caption_file = 'D:/data/vqa/coco/simple_vqa' + '/Annotations/annotations/caption/captions_%s.json'
    question_file = 'D:/data/vqa/coco/simple_vqa' + '/Questions/v2_OpenEnded_mscoco_%s_questions.json'

    vocab_answer_file = 'D:/data/vqa/coco/simple_vqa' + '/vocab_answers.txt'
    answer_dict = text_helper.VocabDict(vocab_answer_file)
    valid_answer_set = set(answer_dict.word_list)

    train = vqa_processing(image_dir, annotation_file,annotation_caption_file, question_file, valid_answer_set, 'train2014')
    valid = vqa_processing(image_dir, annotation_file, annotation_caption_file,question_file, valid_answer_set, 'val2014')
    test = vqa_processing(image_dir, annotation_file, annotation_caption_file,question_file, valid_answer_set, 'test2015')
    test_dev = vqa_processing(image_dir, annotation_file,annotation_caption_file, question_file, valid_answer_set, 'test-dev2015')

    np.save('D:/data/vqa/coco/simple_vqa' + '/train.npy', np.array(train))
    np.save('D:/data/vqa/coco/simple_vqa' + '/valid.npy', np.array(valid))
    np.save('D:/data/vqa/coco/simple_vqa' + '/train_valid.npy', np.array(train + valid))
    np.save('D:/data/vqa/coco/simple_vqa' + '/test.npy', np.array(test))
    np.save('D:/data/vqa/coco/simple_vqa' + '/test-dev.npy', np.array(test_dev))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='/run/media/hoosiki/WareHouse3/mtb/datasets/VQA',
                        help='directory for inputs')

    parser.add_argument('--output_dir', type=str, default='../datasets',
                        help='directory for outputs')

    args = parser.parse_args()

    main(args)
