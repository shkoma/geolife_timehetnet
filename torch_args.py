from datetime import datetime, timedelta
import pandas as pd


global_random = True
global_batch_size = 2000
class ArgumentSet:
    gap = 30
    round_min = str(gap) + 'min'
    round_sec = str(gap) + 's'
    round_time = round_sec

    min = 2
    length = 15 * min
    y_timestep = 5 * min
    sample_s = 2
    sample_q = 1
    batch_size = global_batch_size
    random = global_random
    minimum_dist = 30 # each foot -> 0.1m/1feet (1s) 30m
    max_speed = 50 # normal walking speed: 4.8km/h -> 40/sec, // 6km/h (fast walking) -> 100m/min, 50m/30sec
    # 12km/hour-100m, 30km/hour -> 0.5km/min, 24km/hour: 0.4km/min == 150/10sec,

    train_ratio = 0.9
    output_csv = '_output_' + round_time + '.csv'
    train_csv = '_train_output1_' + round_time + '.csv'
    test_csv = '_test_output1_' + round_time + '.csv'

class ArgumentMask:
    gap = 60
    round_min = str(gap) + 'min'
    round_sec = str(gap) + 's'
    round_time = round_min

    ## Specific Time Grid
    start_time = pd.to_datetime('06:00:00').time()
    finish_time = pd.to_datetime('12:00:00').time()
    time_stamp = finish_time.hour - start_time.hour + 1
    input_day = 2
    output_day = 1
    total_day = input_day + output_day

    random = global_random
    ## DAY GRID ARGUMENT
    # day = 24
    # input_day = 7 * day
    # output_day = int(day//2)
    # day_grid_length = input_day + output_day

    ## CSV ARGUMENT
    train_ratio = 0.9
    output_csv = '_mask_output_' + round_time + '.csv'
    train_csv = '_mask_train_output_' + round_time + '.csv'
    test_csv = '_mask_test_output_' + round_time + '.csv'

class HetnetMask(ArgumentMask):
    sample_s = 2
    sample_q = 1
    total_sample = sample_s + sample_q
    length = ArgumentMask.total_day * ArgumentMask.time_stamp
    y_timestep = ArgumentMask.output_day * ArgumentMask.time_stamp