import numpy as np
import pdb
from plotting.plot import get_mean

h3 = ''' hopper-medium-expert-v2        | 100 scores | score 88.30 +/- 2.86 | return 2853.60 | first value 272.29 | first_search_value 274.48 | step: 809.54 | prediction_error 0.01 | discount_return 254.65
hopper-medium-v2               | 100 scores | score 49.70 +/- 0.90 | return 1597.21 | first value 221.24 | first_search_value 223.37 | step: 499.59 | prediction_error 0.01 | discount_return 230.51
hopper-medium-replay-v2        | 100 scores | score 29.39 +/- 2.83 | return 936.37 | first value 183.83 | first_search_value 188.75 | step: 308.61 | prediction_error 0.01 | discount_return 169.79
walker2d-medium-expert-v2      | 100 scores | score 99.40 +/- 1.66 | return 4564.68 | first value 311.54 | first_search_value 317.21 | step: 962.51 | prediction_error 0.06 | discount_return 305.05
walker2d-medium-v2             | 100 scores | score 65.04 +/- 2.38 | return 2987.43 | first value 211.47 | first_search_value 229.94 | step: 809.61 | prediction_error 0.17 | discount_return 225.53
walker2d-medium-replay-v2      | 100 scores | score 57.58 +/- 2.47 | return 2645.07 | first value 178.24 | first_search_value 198.96 | step: 713.30 | prediction_error 0.23 | discount_return 230.89
halfcheetah-medium-expert-v2   | 100 scores | score 91.74 +/- 0.87 | return 11109.96 | first value 683.52 | first_search_value 713.62 | step: 999.00 | prediction_error 0.18 | discount_return 740.42
halfcheetah-medium-v2          | 100 scores | score 44.01 +/- 0.11 | return 5183.28 | first value 384.78 | first_search_value 395.28 | step: 999.00 | prediction_error 0.29 | discount_return 406.83
halfcheetah-medium-replay-v2   | 100 scores | score 40.59 +/- 0.48 | return 4759.61 | first value 322.86 | first_search_value 345.62 | step: 999.00 | prediction_error 0.57 | discount_return 374.98
ant-medium-expert-v2           | 100 scores | score 126.07 +/- 2.61 | return 4975.91 | first value 438.75 | first_search_value 451.87 | step: 972.75 | prediction_error 0.06 | discount_return 439.94
ant-medium-v2                  | 100 scores | score 87.80 +/- 2.72 | return 3366.45 | first value 314.21 | first_search_value 320.81 | step: 876.75 | prediction_error 0.07 | discount_return 319.85
ant-medium-replay-v2           | 100 scores | score 83.06 +/- 2.75 | return 3167.15 | first value 175.50 | first_search_value 219.35 | step: 895.78 | prediction_error 0.11 | discount_return 289.86'''

h9 = ''' hopper-medium-expert-v2        | 100 scores | score 104.01 +/- 2.09 | return 3364.95 | first value 271.98 | first_search_value 285.07 | step: 917.13 | prediction_error 0.02 | discount_return 268.94
hopper-medium-v2               | 100 scores | score 64.44 +/- 1.42 | return 2076.88 | first value 224.09 | first_search_value 236.59 | step: 625.43 | prediction_error 0.03 | discount_return 242.63
hopper-medium-replay-v2        | 99 scores | score 90.17 +/- 2.03 | return 2914.40 | first value 213.79 | first_search_value 254.32 | step: 898.58 | prediction_error 0.04 | discount_return 251.62
walker2d-medium-expert-v2      | 100 scores | score 107.25 +/- 1.00 | return 4925.22 | first value 311.77 | first_search_value 346.88 | step: 986.83 | prediction_error 0.13 | discount_return 318.76
walker2d-medium-v2             | 100 scores | score 63.21 +/- 2.14 | return 2903.49 | first value 213.05 | first_search_value 266.45 | step: 743.75 | prediction_error 0.74 | discount_return 248.12
walker2d-medium-replay-v2      | 100 scores | score 58.41 +/- 3.40 | return 2682.90 | first value 180.42 | first_search_value 278.86 | step: 646.70 | prediction_error 1.33 | discount_return 243.97
halfcheetah-medium-expert-v2   | 100 scores | score 91.17 +/- 1.03 | return 11038.27 | first value 685.69 | first_search_value 765.35 | step: 999.00 | prediction_error 1.37 | discount_return 746.90
halfcheetah-medium-v2          | 100 scores | score 44.98 +/- 0.11 | return 5303.88 | first value 394.84 | first_search_value 437.58 | step: 999.00 | prediction_error 1.34 | discount_return 418.38
halfcheetah-medium-replay-v2   | 100 scores | score 41.51 +/- 0.42 | return 4873.22 | first value 321.88 | first_search_value 415.96 | step: 999.00 | prediction_error 1.80 | discount_return 388.32
ant-medium-expert-v2           | 100 scores | score 130.96 +/- 2.13 | return 5181.50 | first value 442.04 | first_search_value 491.73 | step: 979.30 | prediction_error 0.21 | discount_return 449.59
ant-medium-v2                  | 100 scores | score 84.36 +/- 3.21 | return 3222.03 | first value 317.84 | first_search_value 351.10 | step: 835.81 | prediction_error 0.34 | discount_return 317.54
ant-medium-replay-v2           | 99 scores | score 98.67 +/- 0.90 | return 3823.70 | first value 164.70 | first_search_value 322.47 | step: 991.91 | prediction_error 0.43 | discount_return 316.94'''

h15 = ''' hopper-medium-expert-v2        | 100 scores | score 105.53 +/- 1.74 | return 3414.17 | first value 272.16 | first_search_value 285.00 | step: 929.79 | prediction_error 0.02 | discount_return 269.40
hopper-medium-v2               | 100 scores | score 63.44 +/- 1.37 | return 2044.42 | first value 224.09 | first_search_value 236.75 | step: 615.27 | prediction_error 0.02 | discount_return 243.13
hopper-medium-replay-v2        | 100 scores | score 87.30 +/- 2.31 | return 2820.81 | first value 213.27 | first_search_value 254.44 | step: 865.10 | prediction_error 0.04 | discount_return 252.87
walker2d-medium-expert-v2      | 100 scores | score 107.44 +/- 0.86 | return 4933.91 | first value 311.73 | first_search_value 346.91 | step: 991.83 | prediction_error 0.13 | discount_return 317.51
walker2d-medium-v2             | 100 scores | score 64.87 +/- 2.10 | return 2979.37 | first value 213.39 | first_search_value 266.74 | step: 758.00 | prediction_error 0.75 | discount_return 248.22
walker2d-medium-replay-v2      | 100 scores | score 66.85 +/- 3.09 | return 3070.36 | first value 178.72 | first_search_value 277.84 | step: 732.33 | prediction_error 1.25 | discount_return 255.33
halfcheetah-medium-expert-v2   | 100 scores | score 91.77 +/- 0.75 | return 11113.80 | first value 685.47 | first_search_value 765.86 | step: 999.00 | prediction_error 1.36 | discount_return 750.08
halfcheetah-medium-v2          | 100 scores | score 45.04 +/- 0.09 | return 5311.08 | first value 395.27 | first_search_value 438.97 | step: 999.00 | prediction_error 1.15 | discount_return 419.99
halfcheetah-medium-replay-v2   | 100 scores | score 40.78 +/- 0.57 | return 4782.61 | first value 328.29 | first_search_value 414.60 | step: 999.00 | prediction_error 1.73 | discount_return 384.25
ant-medium-expert-v2           | 100 scores | score 128.82 +/- 2.36 | return 5091.62 | first value 442.84 | first_search_value 488.94 | step: 970.84 | prediction_error 0.22 | discount_return 442.69
ant-medium-v2                  | 100 scores | score 92.00 +/- 2.38 | return 3543.37 | first value 317.32 | first_search_value 352.48 | step: 897.92 | prediction_error 0.32 | discount_return 335.85
ant-medium-replay-v2           | 100 scores | score 96.71 +/- 1.42 | return 3741.38 | first value 167.43 | first_search_value 322.69 | step: 973.43 | prediction_error 0.43 | discount_return 319.52'''

h21 = ''' hopper-medium-expert-v2        | 100 scores | score 96.40 +/- 2.94 | return 3117.04 | first value 271.81 | first_search_value 301.68 | step: 846.68 | prediction_error 0.19 | discount_return 264.96
hopper-medium-v2               | 100 scores | score 66.70 +/- 1.70 | return 2150.52 | first value 225.64 | first_search_value 248.95 | step: 642.96 | prediction_error 0.15 | discount_return 246.13
hopper-medium-replay-v2        | 100 scores | score 96.35 +/- 1.21 | return 3115.44 | first value 212.90 | first_search_value 278.09 | step: 954.41 | prediction_error 0.12 | discount_return 253.29
walker2d-medium-expert-v2      | 100 scores | score 108.82 +/- 0.85 | return 4997.06 | first value 312.70 | first_search_value 374.65 | step: 991.73 | prediction_error 0.31 | discount_return 323.90
walker2d-medium-v2             | 100 scores | score 52.03 +/- 2.35 | return 2389.96 | first value 209.75 | first_search_value 288.54 | step: 602.35 | prediction_error 2.78 | discount_return 253.76
walker2d-medium-replay-v2      | 100 scores | score 69.25 +/- 3.16 | return 3180.79 | first value 180.40 | first_search_value 309.39 | step: 737.58 | prediction_error 2.80 | discount_return 261.95
halfcheetah-medium-expert-v2   | 100 scores | score 90.60 +/- 0.93 | return 10968.48 | first value 685.80 | first_search_value 843.51 | step: 999.00 | prediction_error 8.08 | discount_return 748.30
halfcheetah-medium-v2          | 100 scores | score 44.70 +/- 0.19 | return 5269.81 | first value 399.31 | first_search_value 482.34 | step: 999.00 | prediction_error 3.94 | discount_return 422.61
halfcheetah-medium-replay-v2   | 100 scores | score 41.36 +/- 0.50 | return 4854.60 | first value 330.33 | first_search_value 465.47 | step: 999.00 | prediction_error 3.92 | discount_return 394.48
ant-medium-expert-v2           | 100 scores | score 134.47 +/- 1.46 | return 5329.22 | first value 442.77 | first_search_value 530.19 | step: 982.93 | prediction_error 0.48 | discount_return 448.48
ant-medium-v2                  | 100 scores | score 89.87 +/- 2.66 | return 3453.84 | first value 319.84 | first_search_value 382.05 | step: 879.46 | prediction_error 0.60 | discount_return 329.32
ant-medium-replay-v2           | 100 scores | score 97.70 +/- 1.27 | return 3782.90 | first value 171.78 | first_search_value 359.24 | step: 973.72 | prediction_error 0.71 | discount_return 316.66'''


def parse(log):
    blocks = log[1:-1].split("\n")
    log_means = dict()
    log_errs = dict()
    for block in blocks:
        chuncks = block.split("|")
        name = chuncks[0].replace(" ", "")
        scores = chuncks[2]
        mean, err = scores.replace("score", "").split(" +/- ")
        log_means[name] = float(mean)
        log_errs[name] = float(err)
    return log_means, log_errs


h3_means, _ = parse(h3)
h9_means, _ = parse(h9)
h15_means, _ = parse(h15)
h21_means, _ = parse(h21)

MEANS = {"Horizon=3": h3_means, "Horizon=9": h9_means, "Horizon=15": h15_means, "Horizon=21": h21_means}
ERRORS = dict()

ALGORITHM_STRINGS = {
}

BUFFER_STRINGS = {
    'medium-expert': 'Medium-Expert',
    'medium': 'Medium',
    'medium-replay': 'Medium-Replay',
}

ENVIRONMENT_STRINGS = {
    'halfcheetah': 'HalfCheetah',
    'hopper': 'Hopper',
    'walker2d': 'Walker2d',
    'ant': 'Ant',
}

SHOW_ERRORS = ['Trajectory\nTransformer', 'Trajectory\nVAE']


def get_result(algorithm, buffer, environment, version='v2'):
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
        return f'${result:.1f}$'


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
    prefix = f'\\multicolumn{{2}}{{c}}{{\\bf Mean ({label})}} & '
    formatted = ' & '.join(str(mean) for mean in means)
    return prefix + formatted


def format_averages_block(algorithms):
    means_filtered = [np.round(get_mean(MEANS[algorithm], exclude='ant'), 1) for algorithm in algorithms]
    means_all = [np.round(get_mean(MEANS[algorithm], exclude=None), 1) for algorithm in algorithms]

    means_all = [
        means
        if 'ant-medium-expert-v2' in MEANS[algorithm]
        else '$-$'
        for algorithm, means in zip(algorithms, means_all)
    ]

    formatted_filtered = format_averages(means_filtered, 'without Ant')
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
        '\\label{table:horizon_scale}\n'
        '\\end{table*}'
    )
    blocks = ''.join(format_buffer_block(algorithms, buffer, environments) for buffer in buffers)
    table = (
        f'{table_prefix}'
        f'{blocks}'
        f'{table_suffix}'
    )
    return table


task_action_dim = {"halfcheetah": 6, "hopper": 3, "walker2d": 6, "ant": 8}

algorithms = ["Horizon=3", "Horizon=9", "Horizon=15", "Horizon=21"]
buffers = ['medium-expert', 'medium', 'medium-replay']
environments = ['halfcheetah', 'hopper', 'walker2d', 'ant']

algo_act_dict = {}
for environment in environments:
    for alg in algorithms:
        for buffer in buffers:
            result = get_result(alg, buffer, environment)
            if isinstance(result, tuple):
                result = result[0]
            elif result == "-":
                result = 0
            else:
                result = result
            if alg not in algo_act_dict:
                algo_act_dict[alg] = {task_action_dim[environment]: [result]}
            elif task_action_dim[environment] not in algo_act_dict:
                algo_act_dict[alg][task_action_dim[environment]] = [result]
            else:
                algo_act_dict[alg][task_action_dim[environment]].append(result)

for k1, v1 in algo_act_dict.items():
    for k2, results in v1.items():
        v1[k2] = np.mean(results)

print(algo_act_dict)

table = format_table(algorithms, buffers, environments)
print(table)


horizon_means = {}

for k1, v1 in MEANS.items():
    all_results = []
    for k2, result in v1.items():
        all_results.append(result)
    horizon_means[int(k1.replace("Horizon=", ""))] = np.mean(all_results)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df = pd.Series(horizon_means)
plt.figure(figsize=(5,5))
plt.subplots_adjust(bottom=0.15, left=0.20)
sns.set_context("paper", font_scale = 2, rc={"lines.linewidth":2})
p = sns.lineplot(data=df, markers=True, markersize=10)
p.set_xlabel("Planning Horizon")
p.set_ylabel("mean score")
plt.ylim(0, 90)
plt.savefig("scalehorizon.png",format='png',dpi=300)


