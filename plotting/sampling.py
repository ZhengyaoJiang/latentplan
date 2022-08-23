import numpy as np
import pdb
from plotting.plot import get_mean

logs={
"beam search":
''' hopper-medium-expert-v2        | 100 scores | score 105.53 +/- 1.74 | return 3414.17 | first value 272.16 | first_search_value 285.00 | step: 929.79 | prediction_error 0.02 | discount_return 269.40
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
ant-medium-replay-v2           | 100 scores | score 96.71 +/- 1.42 | return 3741.38 | first value 167.43 | first_search_value 322.69 | step: 973.43 | prediction_error 0.43 | discount_return 319.52''',
"sample prior":
''' halfcheetah-medium-expert-v2   | 100 scores | score 89.87 +/- 1.24 | return 10877.80 | first value 684.31 | first_search_value 805.99 | step: 999.00 | prediction_error 5.61 | discount_return 742.21
hopper-medium-expert-v2        | 100 scores | score 98.53 +/- 2.46 | return 3186.58 | first value 266.00 | first_search_value 287.70 | step: 871.63 | prediction_error 0.11 | discount_return 265.82
walker2d-medium-expert-v2      | 100 scores | score 107.66 +/- 0.33 | return 4943.94 | first value 311.49 | first_search_value 365.46 | step: 999.00 | prediction_error 0.44 | discount_return 313.28
ant-medium-expert-v2           | 100 scores | score 124.71 +/- 3.04 | return 4918.96 | first value 437.80 | first_search_value 503.84 | step: 956.26 | prediction_error 0.34 | discount_return 437.88
halfcheetah-medium-v2          | 100 scores | score 44.26 +/- 0.51 | return 5214.47 | first value 393.98 | first_search_value 456.41 | step: 999.00 | prediction_error 2.69 | discount_return 412.37
hopper-medium-v2               | 100 scores | score 64.35 +/- 1.65 | return 2074.13 | first value 222.93 | first_search_value 243.15 | step: 626.10 | prediction_error 0.04 | discount_return 241.93
walker2d-medium-v2             | 100 scores | score 55.49 +/- 2.07 | return 2549.07 | first value 209.62 | first_search_value 272.85 | step: 661.57 | prediction_error 1.57 | discount_return 241.16
ant-medium-v2                  | 100 scores | score 88.85 +/- 2.68 | return 3410.76 | first value 313.04 | first_search_value 364.94 | step: 859.13 | prediction_error 0.49 | discount_return 326.67
halfcheetah-medium-replay-v2   | 100 scores | score 39.82 +/- 0.60 | return 4663.59 | first value 298.43 | first_search_value 441.02 | step: 999.00 | prediction_error 2.94 | discount_return 380.31
hopper-medium-replay-v2        | 100 scores | score 79.04 +/- 3.61 | return 2552.00 | first value 183.22 | first_search_value 231.00 | step: 782.38 | prediction_error 0.11 | discount_return 221.53
walker2d-medium-replay-v2      | 100 scores | score 65.96 +/- 3.23 | return 3029.86 | first value 169.62 | first_search_value 266.98 | step: 724.40 | prediction_error 2.05 | discount_return 245.82
ant-medium-replay-v2           | 100 scores | score 81.36 +/- 2.78 | return 3095.99 | first value 167.60 | first_search_value 331.62 | step: 879.49 | prediction_error 0.60 | discount_return 291.19''',
"sample uniform":
''' halfcheetah-medium-expert-v2   | 100 scores | score 41.77 +/- 0.52 | return 4905.63 | first value 438.63 | first_search_value 497.53 | step: 999.00 | prediction_error 4.15 | discount_return 376.33
hopper-medium-expert-v2        | 100 scores | score 62.33 +/- 2.90 | return 2008.17 | first value 249.26 | first_search_value 267.52 | step: 593.97 | prediction_error 0.12 | discount_return 237.16
walker2d-medium-expert-v2      | 100 scores | score 86.74 +/- 1.91 | return 3983.38 | first value 238.56 | first_search_value 274.78 | step: 949.69 | prediction_error 0.63 | discount_return 249.57
ant-medium-expert-v2           | 100 scores | score 105.42 +/- 2.54 | return 4107.78 | first value 351.08 | first_search_value 389.05 | step: 966.28 | prediction_error 0.76 | discount_return 368.66
halfcheetah-medium-v2          | 100 scores | score 39.50 +/- 0.37 | return 4624.10 | first value 367.37 | first_search_value 408.64 | step: 999.00 | prediction_error 4.18 | discount_return 355.34
hopper-medium-v2               | 100 scores | score 39.61 +/- 1.41 | return 1268.97 | first value 219.26 | first_search_value 235.15 | step: 415.55 | prediction_error 0.14 | discount_return 215.01
walker2d-medium-v2             | 100 scores | score 70.24 +/- 1.51 | return 3226.10 | first value 188.92 | first_search_value 228.00 | step: 874.04 | prediction_error 0.93 | discount_return 213.53
ant-medium-v2                  | 100 scores | score 89.80 +/- 2.36 | return 3450.95 | first value 303.84 | first_search_value 338.63 | step: 897.14 | prediction_error 0.85 | discount_return 318.56
halfcheetah-medium-replay-v2   | 100 scores | score 10.18 +/- 0.32 | return 983.86 | first value 172.00 | first_search_value 174.57 | step: 999.00 | prediction_error 3.32 | discount_return 86.01
hopper-medium-replay-v2        | 100 scores | score 14.69 +/- 0.95 | return 457.98 | first value 115.80 | first_search_value 113.66 | step: 185.32 | prediction_error 0.12 | discount_return 166.45
walker2d-medium-replay-v2      | 100 scores | score 7.83 +/- 0.50 | return 360.94 | first value 86.39 | first_search_value 108.73 | step: 180.06 | prediction_error 3.85 | discount_return 127.92
ant-medium-replay-v2           | 100 scores | score 46.96 +/- 1.27 | return 1649.30 | first value 113.21 | first_search_value 104.69 | step: 937.52 | prediction_error 1.43 | discount_return 158.25'''
}


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



MEANS = {f"{k}": parse(v)[0] for k, v in logs.items()}
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
        f'{formatted_filtered} \\\\ \n'
        f'{formatted_all} \\\\ \n'
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
        '\\label{table:ablation_latent_step}\n'
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

algorithms = [f"{k}" for k, v in logs.items() ]
buffers = ['medium-expert', 'medium', 'medium-replay']
environments = ['halfcheetah', 'hopper', 'walker2d', 'ant']


table = format_table(algorithms, buffers, environments)
print(table)

horizon_means = {}

for k1, v1 in MEANS.items():
    all_results = []
    for k2, result in v1.items():
        all_results.append(result)
    horizon_means[int(k1.replace("\\l=", "").replace("$", ""))] = np.mean(all_results)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df = pd.Series(horizon_means)
plt.figure(figsize=(5,5))
plt.subplots_adjust(bottom=0.15, left=0.20)
sns.set_context("paper", font_scale = 2, rc={"lines.linewidth":2})
p = sns.lineplot(data=df, markers=True, markersize=10)
p.set_xlabel("$L$")
p.set_ylabel("mean score")
plt.ylim(0, 90)
plt.savefig("scalelatent.png",format='png',dpi=300)

