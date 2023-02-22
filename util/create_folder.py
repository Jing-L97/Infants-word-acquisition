# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 14:46:56 2022
This is for the data-preprocessing as the input of the Accumulator model
We consider two conditions: with and without morphological change 

@author: Crystal
"""

import os
import pandas as pd
import json
import random

random.seed(10)
MAX_TOKENS=4000

def create_directory(tree_path, metadata_path, text_path, lang, condition):
    metadata = pd.read_csv(metadata_path)
    tree = tree_path + lang + '.json'
    text_path = text_path + lang + '/'+ condition + '/'
    
    # create the corresponding folders
    with open(tree, 'r') as f:
        data = json.load(f)
    
    for i in range(7):
        folder_name = data[0]['contents'][i]['name']
        os.makedirs(lang + '/' + str(folder_name))
        
        for j in range(len(data[0]['contents'][i]['contents'])):
            subfolder_name = data[0]['contents'][i]['contents'][j]['name']
            os.makedirs(lang + '/' + str(folder_name) + '/' + str(subfolder_name))
            # put the files in the corresponding folders
            filename_lst = []
            log_lst = []
            # output the merged file for each folder
            with open(lang + '/' + str(folder_name) + '/' + str(subfolder_name) + '/' + "All_" + condition + '.txt', "w") as outfile:
                for k in range(len(data[0]['contents'][i]['contents'][j]['contents'])):
                    # put the filename in the corresponding folders
                    audio = data[0]['contents'][i]['contents'][j]['contents'][k]['contents'][0]['name']
                    # find the corresponding text files
                    filename = metadata[metadata.book_id == audio]['text_path'].item().split('/')[-1]
                    # merge the files
                    try:
                        with open(text_path + filename, "r") as infile:
                        
                            # Write the contents of the file to the output file
                            outfile.write(infile.read())
                            # Write the filename as a separator between files
                            outfile.write("\n")
                    except:
                        log_lst.append(filename)
                    filename_lst.append(filename)
                    filename_csv = pd.DataFrame(filename_lst) 
                    filename_csv.to_csv(lang + '/'+ str(folder_name) + '/' + str(subfolder_name) + '/' + 'Filename_' + condition + '.csv', index=False, header=False) 
                    log_csv = pd.DataFrame(log_lst) 
                    log_csv.to_csv(lang + '/' + str(folder_name) + '/' + str(subfolder_name) + '/' + "Log_"+ condition + '.csv', index=False, header=False) 
            
            

# output_dir = '/scratch2/jliu/STELAWord/data/formated/'
# lang = "EN"
# condition = 'without'
# text_path = '/scratch2/jliu/STELAWord/data/formated/'
# tree_path = '/scratch1/projects/InfTrain/dataset/trees/'
# metadata_path = '/scratch1/projects/InfTrain/dataset/metadata/matched2.csv'
    
# create_directory(tree_path, metadata_path, text_path, lang, condition)





# split the file into train/test/val
# enumertate the phonetic transcriptions
# Read data
def split_data(tree_path, lang, condition, val_prop, test_prop):
    
    # create the corresponding folders
    tree = tree_path + lang + '.json'
    with open(tree, 'r') as f:
        structure = json.load(f)
    for item in range(7):
        folder_name = structure[0]['contents'][item]['name']
        
        for j in range(len(structure[0]['contents'][item]['contents'])):
            subfolder_name = structure[0]['contents'][item]['contents'][j]['name']
            
            data = []
            for i, line in enumerate(open(lang + '/' + str(folder_name) + '/' + str(subfolder_name) + '/' + "All_" + condition + '.txt', 'r').readlines()):
                splitted = line[:-1].split(' ')
                if len(splitted) >= MAX_TOKENS:
                    # Split list into fixed sized chunks
                    line = [' '.join(splitted[i:i + MAX_TOKENS]) + '\n' for i in range(0, len(splitted), MAX_TOKENS)]
                    data.extend(line)
                else:
                    data.append(line)
            
            size_train = int((1-val_prop - test_prop) * len(data))
            size_val = int(val_prop * len(data))
            data_train, data_val, data_test = data[:size_train], data[size_train:size_train+size_val], data[size_train+size_val:]
            
            data = {'train': data_train,
                    'val': data_val,
                    'test': data_test}
            
            for key, data in data.items():
                output_file = lang + '/' + str(folder_name) + '/' + str(subfolder_name) + '/' + "All_" + condition + '_' + key + '.txt'
                with open(output_file, 'w') as fin:
                    for line in data:
                        fin.write(line)
            print("Done splitting input file into train/dev/test sets.")




# output_dir = '/scratch2/jliu/STELAWord/data/formated/'
# lang = "EN"
# condition = 'without'
# text_path = '/scratch2/jliu/STELAWord/data/formated/'
# tree_path = '/scratch1/projects/InfTrain/dataset/trees/'
# metadata_path = '/scratch1/projects/InfTrain/dataset/metadata/matched2.csv'

# split_data(tree_path, lang, condition, 0.1, 0.1)
