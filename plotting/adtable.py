import numpy as np
import pdb

from plotting.plot import get_mean
from plotting.scores import (
	adroit_means as MEANS,
)
ERRORS = {}
from plotting.parselog import adroit_errs, adroit_means
MEANS["Trajectory\nVAE"] = adroit_means
ERRORS["Trajectory\nVAE"] = adroit_errs

ALGORITHM_STRINGS = {
	'Trajectory\nVAE': 'TAP (Ours)',
	'Trajectory\nTransformer': 'TT',
	'Decision\nTransformer': 'DT',	
}

BUFFER_STRINGS = {
	'expert': 'Expert',
	'cloned': 'Cloned',
	'human': 'Human',
}

ENVIRONMENT_STRINGS = {
	'pen': 'Pen',
	'hammer': 'Hammer',
	'relocate': 'Relocate',
	'door': 'Door',
}

SHOW_ERRORS = ['Trajectory\nVAE']

def get_result(algorithm, buffer, environment, version='v0'):
	key = f'{environment}-{buffer}-{version}'
	mean = MEANS[algorithm].get(key, '-')
	if algorithm in SHOW_ERRORS:
		error = ERRORS[algorithm].get(key)
		return (mean, error)
	else:
		return mean

def format_result(result):
	if type(result) == tuple:
		mean, std = result
		return f'${mean:.1f}$ \\scriptsize{{\\raisebox{{1pt}}{{$\\pm {std:.1f}$}}}}'
	else:
		return f'${result}$'

def format_row(buffer, environment, results):
	buffer_str = BUFFER_STRINGS[buffer]
	environment_str = ENVIRONMENT_STRINGS[environment]
	results_str = ' & '.join(format_result(result) for result in results)
	row = f'{buffer_str} & {environment_str} & {results_str} \\\\ \n'
	return row

def format_buffer_block(algorithms, buffer, environments):
	block_str = '\\midrule\n'
	for environment in environments:
		results = [get_result(alg, buffer, environment) for alg in algorithms]
		row_str = format_row(buffer, environment, results)
		block_str += row_str
	return block_str

def format_algorithm(algorithm):
	algorithm_str = ALGORITHM_STRINGS.get(algorithm, algorithm)
	return f'\multicolumn{{1}}{{c}}{{\\bf {algorithm_str}}}'

def format_algorithms(algorithms):
	return ' & '.join(format_algorithm(algorithm) for algorithm in algorithms)

def format_averages(means, label):
	prefix = f'\\multicolumn{{2}}{{c}}{{\\bf Average ({label})}} & '
	formatted = ' & '.join(str(mean) for mean in means)
	return prefix + formatted

def format_averages_block(algorithms):
	means_filtered = [np.round(get_mean(MEANS[algorithm], exclude='expert'), 1) for algorithm in algorithms]
	means_all = [np.round(get_mean(MEANS[algorithm], exclude=None), 1) for algorithm in algorithms]

	means_all = [
		means
		if 'pen-expert-v0' in MEANS[algorithm]
		else '$-$'
		for algorithm, means in zip(algorithms, means_all)
	]

	formatted_filtered = format_averages(means_filtered, 'without Expert')
	formatted_all = format_averages(means_all, 'all settings')

	formatted_block = (
		f'{formatted_filtered} \\hspace{{.6cm}} \\\\ \n'
		f'{formatted_all} \\hspace{{.6cm}} \\\\ \n'
	)
	return formatted_block

def format_table(algorithms, buffers, environments):
	justify_str = 'll' + 'r' * len(algorithms)
	algorithm_str = format_algorithms(['Dataset', 'Environment'] + algorithms)
	averages_str = format_averages_block(algorithms)
	table_prefix = (
		'\\begin{table*}[h]\n'
		'\\centering\n'
		'\\small\n'
		f'\\begin{{tabular}}{{{justify_str}}}\n'
		'\\toprule\n'
		f'{algorithm_str} \\\\ \n'
	)
	table_suffix = (
		'\\midrule\n'
		f'{averages_str}'
		'\\bottomrule\n'
		'\\end{tabular}\n'
		'\\label{table:d4rl}\n'
		'\\end{table*}'
	)
	blocks = ''.join(format_buffer_block(algorithms, buffer, environments) for buffer in buffers)
	table = (
		f'{table_prefix}'
		f'{blocks}'
		f'{table_suffix}'
	)
	return table


algorithms =['BC', "Opt-MOPO", 'CQL',  'IQL', 'Trajectory\nVAE']
buffers = ['human', 'cloned', 'expert']
environments = ['pen', 'hammer', 'door', 'relocate']

table = format_table(algorithms, buffers, environments)
print(table)
