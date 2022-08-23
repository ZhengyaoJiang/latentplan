import numpy as np
import pdb
from plotting.plot import get_mean

logs={
1:
''' hopper-medium-expert-v2        | 100 scores | score 44.74 +/- 2.57 | return 1435.93 | first value 267.45 | first_search_value 295.50 | step: 391.62 | prediction_error 0.24 | discount_return 254.22
hopper-medium-v2               | 100 scores | score 55.10 +/- 1.82 | return 1772.85 | first value 226.35 | first_search_value 253.58 | step: 519.61 | prediction_error 0.11 | discount_return 255.43
hopper-medium-replay-v2        | 100 scores | score 61.85 +/- 2.61 | return 1992.81 | first value 205.19 | first_search_value 278.27 | step: 594.84 | prediction_error 0.11 | discount_return 253.46
walker2d-medium-expert-v2      | 100 scores | score 94.91 +/- 3.13 | return 4358.46 | first value 311.75 | first_search_value 365.18 | step: 858.64 | prediction_error 0.47 | discount_return 324.35
walker2d-medium-v2             | 100 scores | score 35.19 +/- 2.24 | return 1617.02 | first value 209.49 | first_search_value 294.13 | step: 413.67 | prediction_error 3.36 | discount_return 249.52
walker2d-medium-replay-v2      | 100 scores | score 29.38 +/- 2.00 | return 1350.41 | first value 195.66 | first_search_value 317.30 | step: 344.30 | prediction_error 3.25 | discount_return 235.60
halfcheetah-medium-expert-v2   | 80 scores | score 37.37 +/- 2.89 | return 4359.87 | first value 680.05 | first_search_value 807.27 | step: 999.00 | prediction_error 5.15 | discount_return 400.01
halfcheetah-medium-v2          | 100 scores | score 43.53 +/- 0.19 | return 5123.86 | first value 417.89 | first_search_value 499.46 | step: 999.00 | prediction_error 3.36 | discount_return 403.46
halfcheetah-medium-replay-v2   | 100 scores | score 30.32 +/- 1.24 | return 3484.33 | first value 331.66 | first_search_value 457.06 | step: 999.00 | prediction_error 3.45 | discount_return 312.97
ant-medium-expert-v2           | 100 scores | score 122.69 +/- 2.69 | return 4833.90 | first value 429.37 | first_search_value 507.57 | step: 952.76 | prediction_error 0.48 | discount_return 431.47
ant-medium-v2                  | 100 scores | score 92.28 +/- 2.06 | return 3555.14 | first value 361.70 | first_search_value 394.51 | step: 941.17 | prediction_error 0.72 | discount_return 332.90
ant-medium-replay-v2           | 100 scores | score 80.13 +/- 2.64 | return 3044.04 | first value 257.62 | first_search_value 381.72 | step: 850.15 | prediction_error 0.76 | discount_return 301.54''',
2:
''' halfcheetah-medium-expert-v2   | 80 scores | score 68.12 +/- 2.55 | return 8177.44 | first value 675.47 | first_search_value 815.62 | step: 999.00 | prediction_error 5.07 | discount_return 612.61
hopper-medium-expert-v2        | 60 scores | score 55.62 +/- 4.34 | return 1789.92 | first value 270.67 | first_search_value 299.74 | step: 483.70 | prediction_error 0.20 | discount_return 235.73
walker2d-medium-expert-v2      | 40 scores | score 108.88 +/- 1.37 | return 5000.02 | first value 290.65 | first_search_value 369.26 | step: 981.27 | prediction_error 0.59 | discount_return 324.33
ant-medium-expert-v2           | 60 scores | score 118.01 +/- 4.27 | return 4637.22 | first value 416.28 | first_search_value 514.40 | step: 953.88 | prediction_error 0.51 | discount_return 416.34
halfcheetah-medium-v2          | 80 scores | score 42.57 +/- 0.81 | return 5004.47 | first value 403.23 | first_search_value 469.41 | step: 999.00 | prediction_error 3.00 | discount_return 413.62
hopper-medium-v2               | 80 scores | score 70.47 +/- 1.79 | return 2273.30 | first value 229.52 | first_search_value 246.69 | step: 687.36 | prediction_error 0.04 | discount_return 241.53
walker2d-medium-v2             | 100 scores | score 70.56 +/- 1.92 | return 3240.89 | first value 196.48 | first_search_value 280.85 | step: 829.92 | prediction_error 1.65 | discount_return 246.36
ant-medium-v2                  | 100 scores | score 98.18 +/- 1.86 | return 3803.22 | first value 302.49 | first_search_value 365.48 | step: 957.10 | prediction_error 0.58 | discount_return 339.58
halfcheetah-medium-replay-v2   | 60 scores | score 30.08 +/- 1.50 | return 3453.85 | first value 324.93 | first_search_value 441.15 | step: 999.00 | prediction_error 3.73 | discount_return 335.28
hopper-medium-replay-v2        | 60 scores | score 69.88 +/- 3.78 | return 2253.91 | first value 210.33 | first_search_value 257.15 | step: 688.80 | prediction_error 0.08 | discount_return 246.64
walker2d-medium-replay-v2      | 40 scores | score 60.48 +/- 5.05 | return 2778.17 | first value 146.24 | first_search_value 292.94 | step: 637.00 | prediction_error 1.76 | discount_return 269.95
ant-medium-replay-v2           | 100 scores | score 90.00 +/- 1.92 | return 3459.22 | first value 138.92 | first_search_value 337.61 | step: 935.59 | prediction_error 0.60 | discount_return 308.87''',
3:
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
4:
''' hopper-medium-expert-v2        | 100 scores | score 102.15 +/- 1.89 | return 3304.39 | first value 270.46 | first_search_value 294.02 | step: 905.48 | prediction_error 0.09 | discount_return 267.29
hopper-medium-v2               | 100 scores | score 70.71 +/- 1.80 | return 2280.93 | first value 226.17 | first_search_value 244.16 | step: 686.84 | prediction_error 0.04 | discount_return 242.95
hopper-medium-replay-v2        | 100 scores | score 85.58 +/- 2.49 | return 2764.85 | first value 207.32 | first_search_value 270.42 | step: 840.23 | prediction_error 0.08 | discount_return 251.47
walker2d-medium-expert-v2      | 100 scores | score 108.45 +/- 0.37 | return 4980.37 | first value 312.97 | first_search_value 370.21 | step: 996.04 | prediction_error 0.22 | discount_return 318.80
walker2d-medium-v2             | 100 scores | score 69.33 +/- 1.95 | return 3184.26 | first value 215.14 | first_search_value 282.82 | step: 819.35 | prediction_error 1.16 | discount_return 246.00
walker2d-medium-replay-v2      | 100 scores | score 54.79 +/- 3.84 | return 2516.77 | first value 183.66 | first_search_value 305.11 | step: 602.09 | prediction_error 2.29 | discount_return 233.83
halfcheetah-medium-expert-v2   | 100 scores | score 92.51 +/- 0.13 | return 11204.78 | first value 693.69 | first_search_value 820.13 | step: 999.00 | prediction_error 6.80 | discount_return 742.39
halfcheetah-medium-v2          | 100 scores | score 45.36 +/- 0.10 | return 5351.53 | first value 386.46 | first_search_value 457.73 | step: 999.00 | prediction_error 2.95 | discount_return 423.14
halfcheetah-medium-replay-v2   | 100 scores | score 40.04 +/- 0.81 | return 4690.70 | first value 331.42 | first_search_value 443.39 | step: 999.00 | prediction_error 2.91 | discount_return 374.40
ant-medium-expert-v2           | 100 scores | score 132.23 +/- 1.83 | return 5235.19 | first value 445.74 | first_search_value 511.58 | step: 978.20 | prediction_error 0.35 | discount_return 452.32
ant-medium-v2                  | 100 scores | score 88.90 +/- 2.63 | return 3412.83 | first value 311.42 | first_search_value 360.62 | step: 865.12 | prediction_error 0.44 | discount_return 333.27
ant-medium-replay-v2           | 100 scores | score 95.10 +/- 1.79 | return 3673.59 | first value 241.76 | first_search_value 345.30 | step: 952.79 | prediction_error 0.58 | discount_return 321.16
''',
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



MEANS = {f"$\\l={k}$": parse(v)[0] for k, v in logs.items()}
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

algorithms = [f"$\\l={k}$" for k, v in logs.items() ]
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

