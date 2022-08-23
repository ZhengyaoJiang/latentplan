import numpy as np
import pdb
from plotting.plot import get_mean

logs={
0.0:''' hopper-medium-expert-v2        | 100 scores | score 66.30 +/- 3.00 | return 2137.48 | first value 244.58 | first_search_value 259.05 | step: 636.35 | prediction_error 0.06 | discount_return 237.34
hopper-medium-v2               | 100 scores | score 42.86 +/- 1.12 | return 1374.50 | first value 218.78 | first_search_value 233.57 | step: 442.42 | prediction_error 0.19 | discount_return 220.87
hopper-medium-replay-v2        | 100 scores | score 19.94 +/- 2.50 | return 628.72 | first value 101.78 | first_search_value 111.71 | step: 243.88 | prediction_error 0.16 | discount_return 123.51
walker2d-medium-expert-v2      | 100 scores | score 86.62 +/- 2.55 | return 3978.07 | first value 258.81 | first_search_value 287.76 | step: 933.41 | prediction_error 0.67 | discount_return 254.45
walker2d-medium-v2             | 100 scores | score 63.38 +/- 2.46 | return 2911.25 | first value 195.71 | first_search_value 220.40 | step: 889.83 | prediction_error 0.99 | discount_return 196.40
walker2d-medium-replay-v2      | 100 scores | score 27.08 +/- 2.68 | return 1244.67 | first value 84.10 | first_search_value 94.24 | step: 435.74 | prediction_error 2.92 | discount_return 145.62
halfcheetah-medium-expert-v2   | 100 scores | score 64.31 +/- 2.27 | return 7704.02 | first value 515.31 | first_search_value 589.96 | step: 999.00 | prediction_error 3.84 | discount_return 522.36
halfcheetah-medium-v2          | 100 scores | score 42.57 +/- 0.11 | return 5004.42 | first value 373.74 | first_search_value 426.37 | step: 999.00 | prediction_error 2.30 | discount_return 389.71
halfcheetah-medium-replay-v2   | 100 scores | score 36.25 +/- 0.67 | return 4220.22 | first value 217.01 | first_search_value 270.49 | step: 999.00 | prediction_error 2.56 | discount_return 305.54
ant-medium-expert-v2           | 100 scores | score 104.67 +/- 3.08 | return 4076.15 | first value 357.61 | first_search_value 393.35 | step: 957.59 | prediction_error 0.44 | discount_return 370.20
ant-medium-v2                  | 100 scores | score 85.06 +/- 2.48 | return 3251.61 | first value 304.27 | first_search_value 344.80 | step: 870.90 | prediction_error 0.54 | discount_return 322.61
ant-medium-replay-v2           | 100 scores | score 52.52 +/- 2.54 | return 1883.11 | first value 115.54 | first_search_value 150.04 | step: 853.43 | prediction_error 0.63 | discount_return 202.02''',
0.002:''' hopper-medium-expert-v2        | 100 scores | score 95.13 +/- 2.93 | return 3075.83 | first value 271.55 | first_search_value 297.45 | step: 834.68 | prediction_error 0.08 | discount_return 267.12
hopper-medium-v2               | 100 scores | score 64.56 +/- 1.73 | return 2080.94 | first value 225.94 | first_search_value 248.79 | step: 624.25 | prediction_error 0.11 | discount_return 245.60
hopper-medium-replay-v2        | 100 scores | score 94.44 +/- 1.68 | return 3053.21 | first value 211.28 | first_search_value 272.19 | step: 939.34 | prediction_error 0.07 | discount_return 251.96
walker2d-medium-expert-v2      | 100 scores | score 110.06 +/- 0.07 | return 5054.21 | first value 312.26 | first_search_value 372.91 | step: 999.00 | prediction_error 0.23 | discount_return 324.84
walker2d-medium-v2             | 100 scores | score 49.18 +/- 2.42 | return 2259.13 | first value 209.72 | first_search_value 292.03 | step: 569.20 | prediction_error 2.05 | discount_return 255.08
walker2d-medium-replay-v2      | 100 scores | score 57.31 +/- 3.64 | return 2632.56 | first value 190.64 | first_search_value 303.88 | step: 621.16 | prediction_error 2.49 | discount_return 248.08
halfcheetah-medium-expert-v2   | 100 scores | score 90.96 +/- 0.92 | return 11012.27 | first value 685.44 | first_search_value 811.00 | step: 999.00 | prediction_error 5.89 | discount_return 743.97
halfcheetah-medium-v2          | 100 scores | score 44.76 +/- 0.10 | return 5277.23 | first value 399.59 | first_search_value 465.21 | step: 999.00 | prediction_error 2.81 | discount_return 417.19
halfcheetah-medium-replay-v2   | 100 scores | score 42.17 +/- 0.16 | return 4954.82 | first value 327.05 | first_search_value 446.20 | step: 999.00 | prediction_error 3.05 | discount_return 391.27
ant-medium-expert-v2           | 100 scores | score 132.12 +/- 2.03 | return 5230.26 | first value 442.97 | first_search_value 522.65 | step: 982.74 | prediction_error 0.41 | discount_return 443.44
ant-medium-v2                  | 100 scores | score 88.48 +/- 2.86 | return 3395.28 | first value 321.92 | first_search_value 380.82 | step: 859.13 | prediction_error 0.57 | discount_return 327.07
ant-medium-replay-v2           | 100 scores | score 96.77 +/- 1.45 | return 3743.93 | first value 175.06 | first_search_value 348.40 | step: 968.05 | prediction_error 0.63 | discount_return 315.71''',
0.01:''' hopper-medium-expert-v2        | 100 scores | score 94.53 +/- 2.85 | return 3056.14 | first value 272.14 | first_search_value 297.26 | step: 826.82 | prediction_error 0.09 | discount_return 267.17
hopper-medium-v2               | 100 scores | score 66.99 +/- 1.56 | return 2159.93 | first value 225.91 | first_search_value 248.07 | step: 644.93 | prediction_error 0.08 | discount_return 246.47
hopper-medium-replay-v2        | 100 scores | score 93.17 +/- 1.70 | return 3012.07 | first value 211.45 | first_search_value 272.59 | step: 922.27 | prediction_error 0.06 | discount_return 253.22
walker2d-medium-expert-v2      | 100 scores | score 109.25 +/- 0.75 | return 5016.97 | first value 312.41 | first_search_value 372.39 | step: 992.62 | prediction_error 0.22 | discount_return 324.86
walker2d-medium-v2             | 100 scores | score 53.84 +/- 2.37 | return 2473.45 | first value 209.43 | first_search_value 290.54 | step: 618.37 | prediction_error 1.93 | discount_return 259.43
walker2d-medium-replay-v2      | 100 scores | score 59.56 +/- 3.42 | return 2735.90 | first value 189.13 | first_search_value 304.63 | step: 643.96 | prediction_error 2.59 | discount_return 255.27
halfcheetah-medium-expert-v2   | 100 scores | score 90.26 +/- 1.10 | return 10926.23 | first value 687.49 | first_search_value 811.72 | step: 999.00 | prediction_error 6.04 | discount_return 745.69
halfcheetah-medium-v2          | 100 scores | score 44.80 +/- 0.12 | return 5282.27 | first value 398.16 | first_search_value 464.10 | step: 999.00 | prediction_error 3.02 | discount_return 418.88
halfcheetah-medium-replay-v2   | 100 scores | score 40.52 +/- 0.70 | return 4750.53 | first value 330.73 | first_search_value 446.02 | step: 999.00 | prediction_error 2.96 | discount_return 386.17
ant-medium-expert-v2           | 100 scores | score 133.73 +/- 1.38 | return 5297.99 | first value 442.90 | first_search_value 521.81 | step: 985.36 | prediction_error 0.39 | discount_return 447.88
ant-medium-v2                  | 100 scores | score 89.82 +/- 2.70 | return 3451.78 | first value 321.12 | first_search_value 380.08 | step: 875.82 | prediction_error 0.56 | discount_return 332.12
ant-medium-replay-v2           | 100 scores | score 95.71 +/- 1.82 | return 3699.31 | first value 181.23 | first_search_value 350.64 | step: 957.32 | prediction_error 0.64 | discount_return 312.81''',
0.05:''' hopper-medium-expert-v2        | 100 scores | score 105.53 +/- 1.74 | return 3414.17 | first value 272.16 | first_search_value 285.00 | step: 929.79 | prediction_error 0.02 | discount_return 269.40
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
0.1:''' hopper-medium-expert-v2        | 100 scores | score 106.96 +/- 1.43 | return 3460.86 | first value 272.24 | first_search_value 292.05 | step: 939.83 | prediction_error 0.06 | discount_return 270.32
hopper-medium-v2               | 100 scores | score 63.84 +/- 1.50 | return 2057.42 | first value 224.29 | first_search_value 241.98 | step: 618.96 | prediction_error 0.04 | discount_return 242.75
hopper-medium-replay-v2        | 100 scores | score 97.02 +/- 1.12 | return 3137.17 | first value 212.89 | first_search_value 268.41 | step: 965.75 | prediction_error 0.07 | discount_return 251.96
walker2d-medium-expert-v2      | 100 scores | score 106.50 +/- 1.35 | return 4890.54 | first value 313.34 | first_search_value 362.24 | step: 980.21 | prediction_error 0.20 | discount_return 317.27
walker2d-medium-v2             | 100 scores | score 64.38 +/- 2.14 | return 2957.33 | first value 211.17 | first_search_value 268.84 | step: 746.46 | prediction_error 1.18 | discount_return 249.25
walker2d-medium-replay-v2      | 100 scores | score 65.63 +/- 3.34 | return 3014.26 | first value 185.86 | first_search_value 299.26 | step: 711.52 | prediction_error 2.24 | discount_return 255.56
halfcheetah-medium-expert-v2   | 100 scores | score 92.23 +/- 0.67 | return 11170.78 | first value 685.94 | first_search_value 796.80 | step: 999.00 | prediction_error 5.45 | discount_return 749.25
halfcheetah-medium-v2          | 100 scores | score 45.09 +/- 0.11 | return 5317.89 | first value 393.04 | first_search_value 456.11 | step: 999.00 | prediction_error 2.62 | discount_return 421.51
halfcheetah-medium-replay-v2   | 100 scores | score 40.59 +/- 0.68 | return 4758.85 | first value 320.59 | first_search_value 437.97 | step: 999.00 | prediction_error 2.94 | discount_return 380.79
ant-medium-expert-v2           | 100 scores | score 127.44 +/- 2.61 | return 5033.48 | first value 443.51 | first_search_value 495.61 | step: 959.75 | prediction_error 0.33 | discount_return 446.47
ant-medium-v2                  | 100 scores | score 89.66 +/- 2.73 | return 3444.93 | first value 316.92 | first_search_value 354.77 | step: 870.44 | prediction_error 0.47 | discount_return 322.79
ant-medium-replay-v2           | 100 scores | score 93.66 +/- 1.85 | return 3612.92 | first value 169.81 | first_search_value 339.73 | step: 952.83 | prediction_error 0.58 | discount_return 311.95''',
0.5:''' hopper-medium-expert-v2        | 100 scores | score 76.12 +/- 3.16 | return 2457.09 | first value 221.14 | first_search_value 235.72 | step: 711.00 | prediction_error 0.11 | discount_return 245.50
hopper-medium-v2               | 100 scores | score 47.53 +/- 1.04 | return 1526.51 | first value 220.68 | first_search_value 236.09 | step: 481.58 | prediction_error 0.14 | discount_return 228.67
hopper-medium-replay-v2        | 100 scores | score 2.97 +/- 0.61 | return 76.54 | first value 39.23 | first_search_value 31.17 | step: 33.87 | prediction_error 0.22 | discount_return 40.26
walker2d-medium-expert-v2      | 100 scores | score 104.01 +/- 0.89 | return 4776.27 | first value 310.05 | first_search_value 352.90 | step: 991.70 | prediction_error 0.25 | discount_return 306.17
walker2d-medium-v2             | 100 scores | score 58.67 +/- 2.77 | return 2694.98 | first value 213.30 | first_search_value 217.06 | step: 736.18 | prediction_error 1.24 | discount_return 216.07
walker2d-medium-replay-v2      | 100 scores | score 50.28 +/- 3.24 | return 2309.96 | first value 107.38 | first_search_value 84.51 | step: 651.75 | prediction_error 1.38 | discount_return 191.77
halfcheetah-medium-expert-v2   | 100 scores | score 87.72 +/- 1.41 | return 10609.82 | first value 667.50 | first_search_value 764.19 | step: 999.00 | prediction_error 5.03 | discount_return 718.15
halfcheetah-medium-v2          | 100 scores | score 42.91 +/- 0.50 | return 5047.57 | first value 380.53 | first_search_value 428.35 | step: 999.00 | prediction_error 1.95 | discount_return 398.16
halfcheetah-medium-replay-v2   | 100 scores | score 41.01 +/- 0.22 | return 4811.66 | first value 313.06 | first_search_value 389.87 | step: 999.00 | prediction_error 2.58 | discount_return 364.30
ant-medium-expert-v2           | 100 scores | score 116.21 +/- 3.00 | return 4561.30 | first value 418.33 | first_search_value 456.85 | step: 938.73 | prediction_error 0.36 | discount_return 408.33
ant-medium-v2                  | 100 scores | score 85.11 +/- 2.60 | return 3253.70 | first value 312.43 | first_search_value 343.27 | step: 861.66 | prediction_error 0.43 | discount_return 319.05
ant-medium-replay-v2           | 100 scores | score 76.78 +/- 2.73 | return 2903.20 | first value 160.01 | first_search_value 253.85 | step: 883.45 | prediction_error 0.62 | discount_return 282.89'''
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



MEANS = {f"\\beta={k}": parse(v)[0] for k, v in logs.items()}
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
        '\\label{table:ablation_threshold}\n'
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

algorithms = [f"\\beta={k}" for k, v in logs.items() ]
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
