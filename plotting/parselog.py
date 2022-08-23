gym_log = ''' hopper-medium-expert-v2        | 100 scores | score 105.53 +/- 1.74 | return 3414.17 | first value 272.16 | first_search_value 285.00 | step: 929.79 | prediction_error 0.02 | discount_return 269.40
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

adroit_log = '''
pen-human-v0                   | 100 scores | 76.45 +/- 8.51 
pen-cloned-v0                  | 100 scores | 57.38 +/- 8.68
pen-expert-v0                  | 100 scores | 127.36 +/- 7.72
hammer-human-v0                | 100 scores | 1.43 +/- 0.13
hammer-cloned-v0               | 100 scores | 1.19 +/- 0.08
hammer-expert-v0               | 100 scores | 127.63 +/- 1.70
door-human-v0                  | 100 scores | 8.78 +/- 1.12
door-cloned-v0                 | 100 scores | 11.74 +/- 1.47
door-expert-v0                 | 100 scores | 104.83 +/- 0.84
relocate-human-v0              | 100 scores | 0.25 +/- 0.12
relocate-cloned-v0             | 100 scores | -0.22 +/- 0.01
relocate-expert-v0             | 100 scores | 105.83 +/- 2.69
'''


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

gym_means, gym_errs = parse(gym_log)
adroit_means, adroit_errs = parse(adroit_log)
