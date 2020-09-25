import argparse
import json
import random

import torch
from tqdm import tqdm

from seq2seq import EncoderRNN, AttnDecoderRNN, tensorsFromPair, trainIters, getDevice, evaluate
from seq2seq import Lang, normalizeString

random.seed(7973)


def prepareData(args):
    pairs = []

    if args.phase == 'train_question':
        with open('../executor/data/train.json') as f:
            anns = json.load(f)
        for ann_idx in tqdm(range(len(anns))):
            question_scene = anns[ann_idx]
            for q_idx, q in enumerate(question_scene['questions']):
                question = q['question']
                question = normalizeString(question)
                program = q['program']
                program = ' '.join(program)
                pairs.append([question, program])

    elif args.phase == 'train_choice':
        with open('../executor/data/train.json') as f:
            anns = json.load(f)
        for ann_idx in tqdm(range(len(anns))):
            question_scene = anns[ann_idx]
            for q_idx, q in enumerate(question_scene['questions']):
                if q['question_type'] == 'descriptive': # skip open-ended questions
                    continue
                for c in q['choices']:
                    choice = c['choice']
                    choice = normalizeString(choice)
                    program = c['program']
                    program = ' '.join(program)
                    pairs.append([choice, program])

    print('Total # of pairs: {}'.format(len(pairs)))

    input_lang = Lang('en')
    output_lang = Lang('program')
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words: {} {}; {} {}".format(input_lang.name, input_lang.n_words,
            output_lang.name, output_lang.n_words))

    return input_lang, output_lang, pairs


def train(args):
    input_lang, output_lang, pairs = prepareData(args)
    print(random.choice(pairs))

    model = {}
    model['hidden_size'] = 1000
    model['dropout'] = 0.1
    model['input_lang'] = input_lang
    model['output_lang'] = output_lang
    model['max_length'] = max(input_lang.max_length, output_lang.max_length)+2
    print('Max length: {}'.format(model['max_length']))

    encoder1 = EncoderRNN(input_lang.n_words, model['hidden_size']).to(getDevice())
    encoder1.train()
    attn_decoder1 = AttnDecoderRNN(model['hidden_size'], output_lang.n_words,
            dropout_p=model['dropout'], max_length=model['max_length']).to(getDevice())
    attn_decoder1.train()

    n_iters = 30000
    training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs)) for _ in range(n_iters)]
    trainIters(training_pairs, encoder1, attn_decoder1, n_iters, print_every=1000,
            optim=args.optim, learning_rate=args.learning_rate, max_length=model['max_length'])

    print('saving models...')
    model['encoder_state'] = encoder1.state_dict()
    model['decoder_state'] = attn_decoder1.state_dict()
    torch.save(model, "data/{}_model_checkpoint.pth".format(args.phase.split('_')[-1]))


def inference(args):
    models = {}
    for i in ['question', 'choice']:
        models[i] = torch.load("data/{}_model_checkpoint.pth".format(i))
        models[i]['encoder'] = EncoderRNN(models[i]['input_lang'].n_words, models[i]['hidden_size']).to(getDevice())
        models[i]['encoder'].load_state_dict(models[i]['encoder_state'])
        models[i]['encoder'].eval()
        models[i]['decoder'] = AttnDecoderRNN(models[i]['hidden_size'], models[i]['output_lang'].n_words,
                dropout_p=models[i]['dropout'], max_length=models[i]['max_length']).to(getDevice())
        models[i]['decoder'].load_state_dict(models[i]['decoder_state'])
        models[i]['decoder'].eval()

    with open('../executor/data/validation.json') as f:
        anns = json.load(f)

    out = {}
    for ann in tqdm(anns):
        v = {}
        v['scene_index'] = ann['scene_index']
        v['video_filename'] = ann['video_filename']
        v['questions'] = []

        for ann_q in ann['questions']:
            if ann_q['question_type'] == 'descriptive':
                continue

            q_program_pred, _ = evaluate(models['question']['encoder'], models['question']['decoder'],
                    normalizeString(ann_q['question']),
                    models['question']['input_lang'], models['question']['output_lang'],
                    max_length=models['question']['max_length'])
            if q_program_pred[-1] == '<EOS>':
                q_program_pred = q_program_pred[:-1]

            q = {}
            q['question_program'] = q_program_pred
            q['question'] = ann_q['question']
            q['question_type'] = '{}_multiple_choice'.format(ann_q['question_type'])
            q['question_subtype'] = ann_q['program'][-1]
            q['program_gt'] = ann_q['program']
            q['choices'] = []

            for ann_c in ann_q['choices']:
                c_program_pred, _ = evaluate(models['choice']['encoder'], models['choice']['decoder'],
                        normalizeString(ann_c['choice']),
                        models['choice']['input_lang'], models['choice']['output_lang'],
                        max_length=models['choice']['max_length'])
                if c_program_pred[-1] == '<EOS>':
                    c_program_pred = c_program_pred[:-1]

                c = {}
                c['program'] = c_program_pred
                c['answer'] = ann_c['answer']
                q['choices'].append(c)

            v['questions'].append(q)

        out[v['scene_index']] = v

    out_path = '../executor/parse_results/mc_val_reproduced.json'
    print('Writing output to {}'.format(out_path))
    with open(out_path, 'w') as fout:
        json.dump(out, fout, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['train_question', 'train_choice', 'inference'])
    parser.add_argument('--optim', default='sgd', type=str)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    args = parser.parse_args()

    if args.phase == 'train_question':
        train(args)
    elif args.phase == 'train_choice':
        train(args)
    elif args.phase == 'inference':
        inference(args)
