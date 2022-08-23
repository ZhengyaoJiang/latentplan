import os
import glob
import numpy as np
import json
import pdb

from collections import defaultdict
import latentplan.utils as utils

DATASETS = [
	f'{env}-{buffer}'
	for env in ['hopper', 'walker2d', 'halfcheetah', 'ant']
	for buffer in ['medium-expert-v2', 'medium-v2', 'medium-replay-v2']
]

LOGBASE = os.path.expanduser('~/logs')
TRIAL = '*'
EXP_NAME = 'plans/defaults/freq1_H1_beam50'

def load_results(paths):
	'''
		paths : path to directory containing experiment trials
	'''
	scores = []
	infos = defaultdict(list)
	for i, path in enumerate(sorted(paths)):
		score, info = load_result(path)
		if score is None:
			continue
		scores.append(score)
		for k, v in info.items():
			infos[k].append(v)

		suffix = path.split('/')[-1]

	for k, v in infos.items():
		infos[k] = np.nanmean(v)

	mean = np.mean(scores)
	err = np.std(scores) / np.sqrt(len(scores))
	return mean, err, scores, infos


def load_result(path):
	'''
		path : path to experiment directory; expects `rollout.json` to be in directory
	'''
	#path = os.path.join(path, "0")
	fullpath = os.path.join(path, 'rollout.json')
	suffix = path.split('/')[-1]

	if not os.path.exists(fullpath):
		return None, None

	results = json.load(open(fullpath, 'rb'))
	score = results['score']
	info = dict(returns=results["return"],
				first_value=results["first_value"],
				first_search_value=results["first_search_value"],
                discount_return=results["discount_return"],
				prediction_error=results["prediction_error"],
				step=results["step"])

	return score * 100, info

#######################
######## setup ########
#######################

class Parser(utils.Parser):
	dataset: str = None
	exp_name: str = None

if __name__ == '__main__':

	args = Parser().parse_args()

	for dataset in ([args.dataset] if args.dataset else DATASETS):
		subdirs = glob.glob(os.path.join(LOGBASE, dataset))

		for subdir in subdirs:
			reldir = subdir.split('/')[-1]
			paths = glob.glob(os.path.join(subdir, args.exp_name+"*", TRIAL))

			mean, err, scores, infos = load_results(paths)
			print(f'{dataset.ljust(30)} | {len(scores)} scores | score {mean:.2f} +/- {err:.2f} | '
				  f'return {infos["returns"]:.2f} | first value {infos["first_value"]:.2f} | '
				  f'first_search_value {infos["first_search_value"]:.2f} | step: {infos["step"]:.2f} | '
                  f'prediction_error {infos["prediction_error"]:.2f} | discount_return {infos["discount_return"]:.2f}')
