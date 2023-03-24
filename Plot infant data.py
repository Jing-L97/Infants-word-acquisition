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
import spacy
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

2.1: remove function words in the list
1) POS tagging
2) remove words with vertain POS
Pronouns, prepositions, conjunctions, determiners, qualifiers/intensifiers, and interrogatives

""" 

def match_word(lang, fre_table, mode,selected_freq, word_type):
    infants_data = pd.read_csv(lang + '_' + mode + '.csv')
    # remove annotations in wordbank
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
    
    # select words based on word type
    # get content words
    if lang == 'AE' or lang == 'BE': 
        nlp = spacy.load('en_core_web_sm')
    elif lang == 'FR': 
        nlp = spacy.load('fr_core_news_sm')    
        
    pos_all = []
    for word in selected_words['words']:     
        doc = nlp(word)
        pos_lst = []
        for token in doc:
            pos_lst.append(token.pos_)
        pos_all.append(pos_lst[0])
    selected_words['POS'] = pos_all
    
    func_POS = ['PRON','SCONJ','CONJ','CCONJ','DET','AUX']
    if word_type == 'all':
        selected_words = selected_words
    elif word_type == 'content':
        selected_words = selected_words[~selected_words['POS'].isin(func_POS)]
    elif word_type == 'func':
        selected_words = selected_words[selected_words['POS'].isin(func_POS)]
        
    return selected_words




"""
Step3: get freq bins based on the selected freq above
"""

def chunk_list(lst,group_num):
    n = len(lst)
    chunk_size = int(np.ceil(n/group_num))

    groups = []
    for i in range(0, n, chunk_size):
        group = lst[i:i+chunk_size]
        groups.append(group)

    while len(groups) > group_num:
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
    while len(groups) < group_num:
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

def create_group(lang, mode, selected_words, trial,group_num):
    
    # output the corresponding indices
    final_index_temp = []
    fre_band = []
    
    for num in range(group_num):
        fre_band.append(str(2**num))
        
    for i in range(group_num):  
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

'''
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
'''


def get_statistics(selected_words,lang,mode):
    
    # Calculate median, 5th percentile, and 95th percentile by group
    
    summary = selected_words.groupby('Group').agg({'Norm_freq_per_million': ['count', lambda x: x.quantile(0.05), 'median',  lambda x: x.quantile(0.95)]})
    
    # Rename columns
    
    summary.columns = [('frequency', 'Count'),('frequency', '0.05'), ('frequency', '0.5'),  ('frequency', '0.95')]
    # Reset index and rename columns
    summary = summary.reset_index()
    summary.columns = ['Group', 'Count', '0.05', '0.5', '0.95']
    
    
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




"""
Step4: plot the learning curves
plot the multiple curves in one figure in python, add the labels 
"""

# add the list of values
def plot_curves(lang, selected_words, mode, word_type):
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
    month_lst_transformed = []
    for month in month_lst_final:
        month_lst_transformed.append(int(month))
    # convert into dataframe
    data_frame = pd.DataFrame([month_lst_transformed,size_lst_final,group_lst_final]).T
    data_frame.rename(columns={0:'month',1:'size',2:'Freq band'}, inplace=True) 
    data_frame_final = data_frame.dropna(axis=0)
    sns.set_style('whitegrid')
    ax = sns.lineplot(x="month", y="size", data=data_frame_final, hue="Freq band")
    # set the limits of the x-axis for each line
    for line in ax.lines:
        plt.xlim(0,36)
        plt.ylim(0,1)
        
    if mode == 'comprehension':
        vocab = 'Receptive'
    elif mode == 'production':
        vocab = 'Expressive'
    plt.title(vocab + ' vocabulary of ' + lang + ' ' + word_type + ' words ')
    # display plot
    plt.savefig(lang + '_' + mode + '_developmental data.png')
    plt.show()


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



# also check whether to use all words or only content words
def final_plot(lang,mode,selected_freq, word_type,group_num):
    fre_table, all_words = get_freq(lang)
    selected_words = match_word(lang, fre_table,mode,selected_freq, word_type)
    group_index = chunk_list(selected_words[selected_freq].tolist(),group_num)
    selected_words = create_group(lang, mode, selected_words, group_index,group_num)
    plot_curves(lang, selected_words, mode, word_type)
    stat = get_statistics(selected_words,lang,mode) 
    return stat

word_type = 'content'
selected_freq = 'Log_norm_freq_per_million'
word_type = 'content'  
lang_lst = ['FR','AE','BE']
mode_lst = ['production','comprehension']
group_num = 4

stat_all = pd.DataFrame()
for lang in lang_lst:
    for mode in mode_lst:
        stat = final_plot(lang,mode,selected_freq, word_type,group_num)
        stat_all  = pd.concat([stat_all, stat])  
# concatenete the final statistics
stat_all.to_csv('Stat_all_result.csv')


'''
comparison of different freq bands
box plot on such difference 
'''
df = pd.DataFrame({'Group': ['A', 'A', 'A', 'B', 'B', 'B'],
                   'Value': [10, 20, 30, 40, 50, 60]})
# Create box plot of 'Value' column grouped by 'Group' column
boxplot = df.boxplot(column='Value', by='Group')

# Add title and axis labels
plt.xlabel('Group')
plt.ylabel('Value')

fig = boxplot.figure



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
