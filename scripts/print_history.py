#!/usr/bin/env python
import json, os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('dirs',nargs='*',help='list of model directories to print',default=[])
parser.add_argument('--no-vars',help='supress input variable output',action='store_true')

args = parser.parse_args()

def print_history(directory):
    print(f'Model: {directory}')

    if (os.path.isfile(f'{directory}/input_variables.txt')) and not args.no_vars:
        with open(f'{directory}/input_variables.txt','r') as f_input: input_keys = f_input.read()
        print(f'--Input Variables\n{input_keys}')
    run_files = [ fname for fname in os.listdir(directory) if 'history' in fname and '.json' in fname ]
    def print_run(run_file):
        run_num = run_file.replace('.json','').split('_')[1]
        with open(f'{directory}/{run_file}','r') as f_run: run_data = { key:value['0'] for key,value in json.load(f_run).items() }
        print(f'  [-- Run {run_num} --]'+' - loss: {loss:.4f} - accuracy: {accuracy:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}'.format(**run_data))
    for run_file in run_files: print_run(run_file)
    
for directory in args.dirs: print_history(directory)
