
import argparse
import random
from os import walk
import ast
import pprint
import json
import numpy as np

pp = pprint.PrettyPrinter(indent=4)

def read_pred_data(path,epoch,phase):
    pred_files = []
    
    for (dirpath, dirnames, filenames) in walk(path):
        filenames = list(filter(lambda name: (name.endswith('.pred') and phase in name 
        and not 'agg' in name) == True ,filenames))

        print(filenames)
        if epoch:
            filenames = list(filter(lambda name: ('ep'+ str(epoch) in name) == True ,filenames))
        else:
            filenames.sort(reverse=True)

        pred_files.extend(filenames)
        break
    
    pp.pprint(pred_files)
    num_files = len(pred_files)
    if num_files == 1:
        with open(path + '/' + pred_files[0]) as f:
            data = json.loads(f.read())
        return data, num_files
    
    elif num_files == 2:
        for filename in pred_files:
            if filename.startswith("correct"):
                with open(path + '/' + pred_files[0]) as f:
                    correct_examples = json.loads(f.read())
            else:
                with open(path + '/' + pred_files[0]) as f:
                    incorrect_examples = json.loads(f.read())
        return [correct_examples,incorrect_examples], num_files
    else:
        print("Too many pred files in the dir. Need just 1 or 2.") 
        return
    
    



def shuffle_data(data, data_sub):
    random.shuffle(data)
    
    if not args['data_sub']:
        logger.info("Subsetting: Full dataset selected for prediction.")
        return data
    else:
        if args['data_sub'] < 1:
            assert (args['data_sub'] >= 0)
            sub_val = int(np.floor(len(data)*args['data_sub']))
            data_set = data[:sub_val]
            
        else:
            assert (args['data_sub'] <= len(data))
            data_set = data[:int(args['data_sub'])]
            

    return data_set


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sample random predicted data for analysis.')
    parser.add_argument('--pred_path', help='Directory path of the pred file(s).', required=True)
    parser.add_argument('--data_sub', type=float, help='Fraction of data to subset.', default=None)
    parser.add_argument('--epoch', help='Epoch number for pred file.', type=int, default=None)
    parser.add_argument('--phase', help='Phase of predictions (warmup or joint)', default="joint")
    args = vars(parser.parse_args())

    pp.pprint(args)
    pred_data, num_files = read_pred_data(args['pred_path'],args['epoch'],args['phase'])
    if num_files == 1:
        pred_data = list(pred_data.values())
        sample = shuffle_data(pred_data,args['data_sub'])
        pp.pprint(sample[0])

        sample_path = args['pred_path'] + 'pred_random_sample'
        with open(sample_path, 'w') as file:
            file.write(json.dumps(sample, indent=4))
        print(f"Random Sample Subset saved in file {sample_path}")
    else:
        correct_sample = shuffle_data(list(pred_data[0].values()),args['data_sub'])
        incorrect_sample = shuffle_data(list(pred_data[1].values()),args['data_sub'])
        pp.pprint(correct_sample[0])
        pp.pprint(incorrect_sample[0])

        c_pred_path = args['pred_path'] + '/correct_pred_random_sample'
        i_pred_path = args['pred_path'] + '/incorrect_pred_random_sample'
        with open(c_pred_path, 'w') as file:
            file.write(json.dumps(correct_sample, indent=4))
        print(f"Correct Sample Subset saved in file {c_pred_path}")
        with open(i_pred_path, 'w') as file:
            file.write(json.dumps(incorrect_sample, indent=4))
        print(f"Incorrect Sample Subset saved in file {i_pred_path}")
