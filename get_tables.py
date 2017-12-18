from helpers import load_obj
import argparse
from os import listdir
from os.path import isfile, join
from sklearn.metrics import f1_score

# Example usage:
# python get_tables.py -p ./results_train_test -o results/lambda_notune.txt -s _results

##########################
#####   PARSING       ####
##########################

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--results_path",
                    default='./results_train_test',
                    help=("full path to the result"))
parser.add_argument("-o", "--out_name",
                    required=True,
                    help=("full path and filename to outfile"))
parser.add_argument("-s", "--substring",
                    required=True,
                    help=("substring that must be contained in\
                            the pkls to be processed"))
parser.add_argument("-d", "--del_substring",
                    default="thereisnowaythisubstringwouldexist",
                    help=("substring that must not be contained in\
                            the pkls to be processed"))

##########################
#####     INIT        ####
##########################

### Command line arguments
args = vars(parser.parse_args())
print(args)
results_path = args['results_path']
substring = args['substring']
del_substring = args['del_substring']
out_name = args['out_name']

### Get the result files
files = [join(results_path, f) for f in listdir(results_path) \
        if isfile(join(results_path, f)) and (substring in f) and (del_substring not in f)]
print(files)

with open(out_name,"w") as outfile:
    for filename in files:
        result = load_obj(filename)
        line = []
        line.append(filename)
        line.append('%.02f'%result['train_accuracy'])
        line.append('%.02f'%result['test_accuracy'])
        f1=f1_score(result['test_true_classes'], result['test_pred_classes'], average='weighted')
        line.append('%.02f'%f1)
        for elem in line:
            outfile.write(elem)
            outfile.write(" & ")
        outfile.write("\n")
