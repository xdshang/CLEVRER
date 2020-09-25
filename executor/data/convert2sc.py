import json
import argparse
from collections import defaultdict

from tqdm import tqdm


def inspect_template():
    with open('train.json') as fin:
        anns = json.load(fin)
    with open('validation.json') as fin:
        anns.extend(json.load(fin))

    questions = defaultdict(set)
    for ann in anns:
        for q in ann['questions']:
            q_type = q['question_type']
            if q_type == 'descriptive':
                continue
            elif q_type == 'counterfactual':
                prefix = q['question'].split(',')[-1].lower()
                prefix = prefix.split()[:6]
            else:
                prefix = q['question'].split()[:6]
            questions[q_type].add(' '.join(prefix))
    
    for q_type, prefixes in questions.items():
        print('{}:'.format(q_type))
        for prefix in sorted(prefixes):
            print('\t{}'.format(prefix))


def convert(split):
    with open('{}.json'.format(split)) as fin:
        anns = json.load(fin)

    num_total = 0
    num_yes = 0
    out = []
    for ann in anns:
        v = dict(ann)
        v['questions'] = []

        for ann_q in ann['questions']:
            if ann_q['question_type'] == 'descriptive':
                continue
            for ann_c in ann_q['choices']:
                q = dict(ann_q)
                del q['choices']
                # merge question and choice
                question = ann_q['question'].lower()
                # predictive
                if 'what will happen next' in question:
                    merged_q = question.replace('what will happen next',
                                                'will {} next'.format(ann_c['choice']))
                    merged_q = merged_q.replace('collides', 'collide')
                elif 'which event will happen next' in question:
                    merged_q = question.replace('which event will happen next',
                                                'will {} next'.format(ann_c['choice']))
                    merged_q = merged_q.replace('collides', 'collide')
                # explanatory
                elif 'which of the following is responsible for' in question: 
                    merged_q = question.replace('which of the following is responsible for',
                                                'is {} responsible for'.format(ann_c['choice']))
                elif 'which of the following is not responsible for' in question:
                    merged_q = question.replace('which of the following is not responsible for',
                                                'is {} not responsible for'.format(ann_c['choice']))
                # counterfactual
                elif 'what will happen' in question:
                    merged_q = question.replace('what will happen',
                                                'will {}'.format(ann_c['choice']))
                    merged_q = merged_q.replace('collides', 'collide')
                elif 'what will not happen' in question:
                    merged_q = question.replace('what will not happen',
                                                'will {}'.format(ann_c['choice']))
                    merged_q = merged_q.replace('collides', 'collide')
                    merged_q = merged_q.replace('collide', 'not collide')
                elif 'which event will happen' in question:
                    merged_q = question.replace('which event will happen',
                                                'will {}'.format(ann_c['choice']))
                    merged_q = merged_q.replace('collides', 'collide')
                elif 'which event will not happen' in question:
                    merged_q = question.replace('which event will not happen',
                                                'will {}'.format(ann_c['choice']))
                    merged_q = merged_q.replace('collides', 'collide')
                    merged_q = merged_q.replace('collide', 'not collide')
                elif 'which of the following will happen' in question:
                    merged_q = question.replace('which of the following will happen',
                                                'will {}'.format(ann_c['choice']))
                    merged_q = merged_q.replace('collides', 'collide')
                elif 'which of the following will not happen' in question:
                    merged_q = question.replace('which of the following will not happen',
                                                'will {}'.format(ann_c['choice']))
                    merged_q = merged_q.replace('collides', 'collide')
                    merged_q = merged_q.replace('collide', 'not collide')
                else:
                    raise Exception('Cannot match any template: {}'.format(question))
                q['question'] = merged_q[0].upper()+merged_q[1:].lower()

                # merge programs
                q['program'] = ann_c['program'] + ann_q['program']

                if ann_c['answer'] == 'correct':
                    q['answer'] = 'yes'
                    num_yes += 1
                else:
                    q['answer'] = 'no'
                num_total += 1

                v['questions'].append(q)
        
        out.append(v)

    print('Ratio of "yes" in {}: {:.2f}%'.format(split, 100.0*num_yes/num_total))
    with open('../parse_results/sc_{}.json'.format(split), 'w') as fout:
        json.dump(out, fout, separators=(',', ':'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert multi-choice question to multiple single-choice questions')
    parser.add_argument('--phase', required=True, choices=['inspect', 'convert'])
    args = parser.parse_args()

    if args.phase == 'inspect':
        inspect_template()
    elif args.phase == 'convert':
        convert('train')
        convert('validation')
