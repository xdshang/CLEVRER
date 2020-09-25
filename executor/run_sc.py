"""
Run symbolic reasoning on open-ended questions
"""
import os
import json
from tqdm import tqdm
import argparse

from executor import Executor
from simulation import Simulation


parser = argparse.ArgumentParser()
parser.add_argument('--n_progs', required=True)
parser.add_argument('--use_event_ann', default=1, type=int)
parser.add_argument('--use_in', default=0, type=int)  # Interaction network for dynamics prediction
args = parser.parse_args()



if args.use_event_ann != 0:
    raw_motion_dir = 'data/propnet_preds/with_edge_supervision'
else:
    raw_motion_dir = 'data/propnet_preds/without_edge_supervision'
if args.use_in:
    raw_motion_dir = 'data/propnet_preds/interaction_network'

question_path = './parse_results/sc_validation.json'
program_path = './parse_results/sc_val_{}.json'.format(args.n_progs)

with open(program_path) as f:
    parsed_pgs = json.load(f)
with open(question_path) as f:
    anns = json.load(f)

total, correct = 0, 0
total_expl, correct_expl = 0, 0
total_pred, correct_pred = 0, 0
total_coun, correct_coun = 0, 0

pbar = tqdm(range(5000))

for ann_idx in pbar:
    question_scene = anns[ann_idx]
    file_idx = ann_idx + 10000 
    ann_path = os.path.join(raw_motion_dir, 'sim_%05d.json' % file_idx)

    sim = Simulation(ann_path, use_event_ann=(args.use_event_ann != 0))
    exe = Executor(sim)

    for q_idx, q in enumerate(question_scene['questions']):
        q_type = q['question_type']
        if q_type == 'descriptive':
            continue
        question = q['question']
        parsed_pg = parsed_pgs[str(file_idx)]['questions'][q_idx]['question_program']
        pred = exe.run(parsed_pg, debug=False)
        ans = q['answer']
        if pred == ans:
            correct += 1
        total += 1
        if q_type.startswith('explanatory'):
            correct_expl += pred == ans
            total_expl += 1
        elif q_type.startswith('predictive'):
            correct_pred += pred == ans
            total_pred += 1
        elif q_type.startswith('counterfactual'):
            correct_coun += pred == ans
            total_coun += 1

    pbar.set_description('acc: {:f}%%'.format(float(correct)*100/total))

print('============ results ============')
print('overall accuracy per question: %f %%' % (float(correct) * 100.0 / total))
print('explanatory accuracy per question: %f %%' % (float(correct_expl) * 100.0 / total_expl))
print('predictive accuracy per question: %f %%' % (float(correct_pred) * 100.0 / total_pred))
print('counterfactual accuracy per question: %f %%' % (float(correct_coun) * 100.0 / total_coun))
print('============ results ============')

output_ann = {
    'total_question': total,
    'correct_question': correct,
    'total_explanatory_question': total_expl,
    'correct_explanatory_question': correct_expl,
    'total_predictive_question': total_pred,
    'correct_predictive_question': correct_pred,
    'total_counterfactual_question': total_coun,
    'correct_counterfactual_question': correct_coun,
}

output_file = 'result_sc.json'
if args.use_in != 0:
    output_file = 'result_sc_in.json'
with open(output_file, 'w') as fout:
    json.dump(output_ann, fout)