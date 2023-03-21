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
    fre_table['Norm_freq_per_million'] = fre_table['Norm_freq']*1000000
    # get log_freq
    log_freq_lst = []
    for freq in freq_lst:
        log_freq = math.log2(freq)
        log_freq_lst.append(log_freq)
    fre_table['Log_freq'] = log_freq_lst
    
    # get logarithm of normalized word freq per million
    norm_log_freq_lst = []
    for freq in fre_table['Norm_freq_per_million'].tolist():
        norm_log_freq = math.log2(freq)
        norm_log_freq_lst.append(norm_log_freq)
    fre_table['Log_norm_freq_per_million'] = norm_log_freq_lst
    
    fre_table.to_csv(filepath + '_Freq_all.csv')
    return fre_table, result


"""
Step2: match the word freq with the infants data
mode: 'compre' or 'prod'
select the freq to group freq bins
"""
def match_word(lang, fre_table, mode,selected_freq):
    infants_data = pd.read_csv(lang + '_' + mode + '.csv')
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
    log_freq_lst = []
    norm_freq_lst = []
    n = 0
    while n < fre_table["Word"].shape[0]:
        selected_rows = infants_data[infants_data['words'] == fre_table["Word"].tolist()[n]] 
        
        if selected_rows.shape[0] > 0:
            # for polysemies, only take the first meaning; OR 
            clean_selected_rows = infants_data[infants_data['words'] == fre_table["Word"].tolist()[n]].head(1)
            
            log_freq_lst.append(fre_table['Log_norm_freq_per_million'].tolist()[n])
            norm_freq_lst.append(fre_table['Norm_freq_per_million'].tolist()[n])
            df = pd.concat([df, clean_selected_rows])    
        n += 1
        
    df['Log_norm_freq_per_million'] = log_freq_lst
    df['Norm_freq_per_million'] = norm_freq_lst
    selected_words = df.sort_values(selected_freq)
    return selected_words



"""
Step3: get freq bins based on the selected freq above
"""

def chunk_list(lst):
    n = len(lst)
    chunk_size = int(np.ceil(n/6))

    groups = []
    for i in range(0, n, chunk_size):
        group = lst[i:i+chunk_size]
        groups.append(group)

    while len(groups) > 6:
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
    while len(groups) < 6:
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

def create_group(lang, mode, selected_words, trial):
    
    # output the corresponding indices
    final_index_temp = []
    fre_band = ['1', '2','4', '16','32', '64' ]
    for i in range(6):  
        final_index_temp.append([fre_band[i]] * len(trial[i]))
        
    
    # flatten the list
    final_index_lst = [item for sublist in final_index_temp for item in sublist]    
    selected_words['Group'] = final_index_lst
    selected_words.to_csv(lang + '_' + mode + '_' + 'freq_selected.csv')
    return selected_words

"""
get statistics of the results
medians, quartiles
"""   


def get_statistics(selected_words,lang,mode):
    
    
    # Calculate median, 5th percentile, and 95th percentile by group
    summary = selected_words.groupby('Group')['Norm_freq_per_million'].quantile([0.5, 0.05, 0.95])
    
    # Rename columns for readability
    summary = summary.reset_index().rename(columns={'level_1': 'Percentile'})
    # Pivot table to make percentiles columns
    summary = summary.pivot(index='Group', columns='Percentile', values='Norm_freq_per_million')
    summary['Lang'] = lang
    summary['Aspect'] = mode
    
    summary = summary.reset_index()
    # convert into int form for sorting
    group_lst = []
    for group in summary['Group'].tolist():
        group_lst.append(int(group))
    summary['Group'] = group_lst
    
    sorted_summary = summary.sort_values('Group')
    sorted_summary.to_csv(lang + '_' + mode + '_' + 'stat.csv')
    return sorted_summary

lang = 'FR'
mode = 'prod'
selected_words = pd.read_csv('FR_prod_freq_selected.csv')

get_statistics(selected_words,lang,mode)

"""
Step4: plot the learning curves
plot the multiple curves in one figure in python, add the labels 
"""

# add the list of values
def plot_curves(lang, selected_words, mode):
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
    data_frame.rename(columns={0:'month',1:'size',2:'Freq band'}, inplace=True) 
    data_frame_final = data_frame.dropna(axis=0)
    sns.set_style('whitegrid')
    sns.lineplot(data=data_frame_final, x="month", y="size", hue="Freq band")
    plt.title(lang + ' ' + mode)
    plt.savefig(lang + '_' + mode + '_developmental data.png', bbox_inches='tight')


def plot_slope(lang, mode, selected_words):
    slope_lst = []
    grouped_words = selected_words.groupby(['Group']).mean()
    updated = grouped_words.reset_index()
    freq_lst = []
    for num in updated['Group'].tolist():
        freq_lst.append(int(num))
    n = 0
    while n < updated.shape[0]:
        if lang == 'BE':
            slope = updated['25'].tolist()[n] - updated['12'].tolist()[n]
        elif lang == 'AE':
            slope = updated['18'].tolist()[n] - updated['8'].tolist()[n]
        elif lang == 'FR':
            slope = updated['16'].tolist()[n] - updated['8'].tolist()[n]    
        slope_lst.append(slope)
        n += 1  
    updated['Vocab size change'] = slope_lst 
    updated['freq'] = freq_lst 
    sns.lineplot(data=updated, x='freq', y="Vocab size change")
    plt.title(lang + mode)




def final_plot(lang,mode,selected_freq):
    fre_table, all_words = get_freq(lang)
    selected_words = match_word(lang, fre_table,mode,selected_freq)
    group_index = chunk_list(selected_words[selected_freq].tolist())
    selected_words = create_group(lang, mode, selected_words, group_index)
    #plot_curves(lang, selected_words, mode)
    stat = get_statistics(selected_words,lang,mode) 
    return stat

selected_freq = 'Log_norm_freq_per_million'
lang_lst = ['FR','AE','BE']
mode_lst = ['prod','compre']
for lang in lang_lst:
    for mode in mode_lst:
        final_plot(lang,mode,selected_freq)
    
 
df = pd.DataFrame({'Group': ['A', 'A', 'A', 'B', 'B', 'B'],
                   'Value': [10, 20, 30, 40, 50, 60]})


'''
comparison of different freq bands
'''

# Create box plot of 'Value' column grouped by 'Group' column
boxplot = df.boxplot(column='Value', by='Group')

# Add title and axis labels
plt.xlabel('Group')
plt.ylabel('Value')

fig = boxplot.figure



import pandas as pd
import matplotlib.pyplot as plt

# Create example dataframe
df = pd.DataFrame({'values': [1, 2, 3, 4, 5, 6]})

# Calculate statistics
median = df['values'].median()
q5 = df['values'].quantile(0.05)
q95 = df['values'].quantile(0.95)

# Draw box plot
plt.boxplot(df['values'], positions=[1])
plt.plot([0.75, 1.25], [median, median], 'k-', linewidth=2)
plt.plot([1, 1], [q5, q95], 'k-', linewidth=2)
plt.plot([0.75, 1.25], [q5, q5], 'k--', linewidth=1)
plt.plot([0.75, 1.25], [q95, q95], 'k--', linewidth=1)
plt.xticks([1], ['Values'])
plt.show()
