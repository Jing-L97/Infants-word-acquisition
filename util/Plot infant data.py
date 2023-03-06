# -*- coding: utf-8 -*-
"""
plot the freq curve
"""
import string
import os
import pandas as pd
import collections
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
"""
Step1: count freq
"""

def count_words(path):       
    with open(path) as f:
        file = f.readlines()
        cleaned_lst = []
        for script in file: 
            if script.startswith('*') and not script.startswith('*CHI'):
                translator = str.maketrans('', '', string.punctuation+ string.digits)
                clean_string = script.translate(translator).lower()
                cleaned_lst.append(clean_string)
                
    trial = pd.DataFrame(cleaned_lst, columns = ['whole'])
    trial[['speaker', 'content']] = trial['whole'].str.split('\t', expand=True)
    trial_lst = trial['content'].tolist()
    result_lst = []
    for i in trial_lst:
        removed_multi = re.sub(r"\s+", " ", i)
        removed_start = removed_multi.lstrip()
        removed_end = removed_start.rstrip() 
        result = removed_end.split(' ')
        
        for j in result:
            cleaned_word = j.strip()
            if len(cleaned_word) > 0:  
                result_lst.append(cleaned_word)
    
    return result_lst
    
    
    
def get_freq(filepath):
# Load the .cha file
    
    # three folder structures
    final = [] 
    for file in os.listdir(filepath):
        foldername = os.fsdecode(file)
        
        for folder in os.listdir(filepath + '/' + foldername):
            subfolder = os.fsdecode(folder)
            
            for subfoldername in os.listdir(filepath + '/' + foldername + '/' + subfolder):
                textfolder = os.fsdecode(subfoldername)
                
                try:
                    for text in os.listdir(filepath + '/' + foldername + '/' + subfolder+ '/' + textfolder):
                        textname = os.fsdecode(text)
                        
                        for file in os.listdir(filepath + '/' + foldername + '/' + subfolder+ '/' + textfolder + '/' +textname):
                            filename = os.fsdecode(file)
                            CHApath = filepath + '/' + foldername + '/' + subfolder+ '/' + textfolder + '/' + textname + '/' + filename
                            result_lst = count_words(CHApath)
                            final.append(result_lst)
                except:
                    pass
                        
    
    # flatten the list
    result = [item for sublist in final for item in sublist]
    frequencyDict = collections.Counter(result)  
    freq_lst = list(frequencyDict.values())
    word_lst = list(frequencyDict.keys())
    
    # get freq
    fre_table = pd.DataFrame([word_lst,freq_lst]).T
    col_Names=["Word", "Freq"]
    fre_table.columns = col_Names
    fre_table['Norm_freq'] = fre_table['Freq']/len(result)
    # get log_freq
    log_freq_lst = []
    for freq in freq_lst:
        log_freq = math.log2(freq)
        log_freq_lst.append(log_freq)
    fre_table['Log_freq'] = log_freq_lst
    fre_table.to_csv(filepath + '_Freq_all.csv')
    return fre_table, result


"""
Step2: match the word freq with the infants data
"""
def match_word(lang, fre_table):
    infants_data = pd.read_csv(lang + '_compre.csv')
    # remove annotations
    words = infants_data['item_definition'].tolist()
    cleaned_lst = []
    for word in words:
        # remove punctuations
        translator = str.maketrans('', '', string.punctuation+ string.digits)
        clean_string = word.translate(translator).lower()
        # remove annotations; problem: polysemies
        cleaned_word = re.sub(r"\([a-z]+\)", "",clean_string)
        cleaned_lst.append(cleaned_word)
    
    infants_data['words'] = cleaned_lst
    # merge similar words
    # match two columns
    # merge dataframes based on columns 'Word' and 'words'
    df = pd.DataFrame()
    freq_lst = []
    n = 0
    while n < fre_table["Word"].shape[0]:
        selected_rows = infants_data[infants_data['words'] == fre_table["Word"].tolist()[n]] 
        
        if selected_rows.shape[0] > 0:
            # for polysemies, only take the first meaning; OR 
            clean_selected_rows = infants_data[infants_data['words'] == fre_table["Word"].tolist()[n]].head(1)
            
            freq_lst.append(fre_table["Log_freq"].tolist()[n])
            df = pd.concat([df, clean_selected_rows])    
        n += 1
        
    df['Log_freq'] = freq_lst
    selected_words = df.sort_values('Log_freq')
    return selected_words





"""
Step3: get freq bins
"""

def chunk_list(lst):
    n = len(lst)
    chunk_size = int(np.ceil(n/7))

    groups = []
    for i in range(0, n, chunk_size):
        group = lst[i:i+chunk_size]
        groups.append(group)

    while len(groups) > 7:
        new_groups = []
        for i in range(0, len(groups)-1, 2):
            merged_group = groups[i] + groups[i+1]
            new_group_size = int(np.ceil(len(merged_group)/2))
            new_groups.extend([merged_group[:new_group_size], merged_group[new_group_size:]])
        if len(groups) % 2 != 0:
            new_groups.append(groups[-1])
        groups = new_groups

    medians = [np.median(group) for group in groups]
    median_diffs = [abs(medians[i+1] - medians[i]) for i in range(len(medians)-1)]
    while len(groups) < 7:
        max_diff_index = median_diffs.index(max(median_diffs))
        split_group = groups[max_diff_index]
        new_group_size = int(np.ceil(len(split_group)/2))
        groups[max_diff_index:max_diff_index+1] = [split_group[:new_group_size], split_group[new_group_size:]]
        medians = [np.median(group) for group in groups]
        median_diffs = [abs(medians[i+1] - medians[i]) for i in range(len(medians)-1)]

    return groups



# chunk into different frequency bins
"""
use log freq bins to create different subgroups
"""        

def create_group(selected_words, trial):
    
    # output the corresponding indices
    final_index_temp = []
    for i in range(7):  
        final_index_temp.append([i] * len(trial[i]))
        
    # final_index_temp = []
    # label_lst = ['1st','2nd','4th','8th','16th','32th','64th']
    # for i in range(7):
    #     final_index_temp.append([label_lst[i]] * len(final[i]))
    
    # flatten the list
    final_index_lst = [item for sublist in final_index_temp for item in sublist]    
    selected_words['Group'] = final_index_lst
    selected_words.to_csv('Freq_selected.csv')
    return selected_words



"""
Step4: plot the learning curves
plot the multiple curves in one figure in python, add the labels 
"""

# add the list of values
def plot_curves(lang, selected_words):
    size_lst = []
    month_lst = []
    group_lst = []
    n = 0
    while n < selected_words.shape[0]:
        
        if lang == 'FR':
            size_lst.append(selected_words.iloc[n].tolist()[4:12])
            group = selected_words['Group'].tolist()[n]
            group_lst.append([group] * 8)
            
            headers_list = selected_words.columns.tolist()[4:12]
        elif lang == 'AE':
            size_lst.append(selected_words.iloc[n].tolist()[4:15])
            group = selected_words['Group'].tolist()[n]
            group_lst.append([group] * 11)
            headers_list = selected_words.columns.tolist()[4:15]
        
        elif lang == 'BE':
            size_lst.append(selected_words.iloc[n].tolist()[4:18])
            group = selected_words['Group'].tolist()[n]
            group_lst.append([group] * 14)
            headers_list = selected_words.columns.tolist()[4:18]
            
        month_lst.append(headers_list)
        n += 1
    size_lst_final = [item for sublist in size_lst for item in sublist]
    month_lst_final = [item for sublist in month_lst for item in sublist]
    group_lst_final = [item for sublist in group_lst for item in sublist]
    # convert into dataframe
    data_frame = pd.DataFrame([month_lst_final,size_lst_final,group_lst_final]).T
    data_frame.rename(columns={0:'month',1:'size',2:'group'}, inplace=True) 
    data_frame_final = data_frame.dropna(axis=0)
    sns.lineplot(data=data_frame_final, x="month", y="size", hue="group")
    plt.title(lang)
    plt.savefig(lang + '_developmental data.png', bbox_inches='tight')


def final_plot(lang):
    fre_table = get_freq(lang)
    selected_words = match_word(lang, fre_table)
    trial = chunk_list(selected_words['Log_freq'].tolist())
    fre_table = get_freq(lang)
    selected_words = create_group(selected_words, trial)
    plot_curves(lang, selected_words)




