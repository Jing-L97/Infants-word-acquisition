# -*- coding: utf-8 -*-
"""
plot the freq curve
"""

import os
import pandas as pd
import string
import numpy as np
import math
import sys
sys.setrecursionlimit(10000)


"""
Step1: put all the results in one csv file

/scratch1/projects/InfTrain/models/EN/50h/00/lstm/scores/swuggy/en/swuggy_en_testset_1
"""
def get_data(root_dir,lang):
    df = pd.DataFrame()
    hour_lst = []
    chunk_lst = []
    freq_lst = []
    weighted_mean_lst = []
    # iterate all the folders
    for file in os.listdir(root_dir + lang):
        # iterate hours
        hour_name = os.fsdecode(file)
        try:
            for folder in os.listdir(root_dir + lang + '/' + hour_name):
                # iterate chunks
                chunk_name = os.fsdecode(folder)
                
                for subfoldername in os.listdir(root_dir + lang + '/' + hour_name + '/' + chunk_name +'/lstm/scores/swuggy/' + lang.lower()):
                    textfolder = os.fsdecode(subfoldername)
                    # read .csv file
                    result = pd.read_csv(root_dir + lang + '/' + hour_name + '/' + chunk_name +'/lstm/scores/swuggy/' + lang.lower() + '/' + textfolder + '/score_lexical_dev_by_frequency.csv')           
                    
                    
                    df = pd.concat([df, result])  
                    n = 0
                    while n < result.shape[0]:
                        # put the hour in a list
                        hour_lst.append(hour_name[:-1])
                        # put the chunk in a list
                        chunk_lst.append(chunk_name)
                        # put the freq band in a list
                        freq_lst.append(textfolder.split('_')[-1])
                        n += 1
                    
                    # get weighted average
                    result.dropna(inplace=True)
                    total = 0
                    n_all = 0
                    std = 0
                    m = 0
                    # in the case of the non-empty frame
                    if result.shape[0] > 0:
                        while m < result.shape[0]:
                            total += result['n'].tolist()[m] * result['score'].tolist()[m]
                            n_all += result['n'].tolist()[m]
                            std += result['std'].tolist()[m]
                            m += 1
                        weighted_mean_lst.append([total/n_all,std/result.shape[0],hour_name,chunk_name,textfolder.split('_')[-1]])
                        
        except:
           print([folder,subfoldername]) 
    
    df['hour'] = hour_lst
    df['chunk'] = chunk_lst
    df['freq'] = freq_lst
    df.to_csv(lang + '_Raw.csv')            
    
    
    mean_frame = pd.DataFrame(weighted_mean_lst, columns=['Lexical score', 'mean_std','Quantity of speech (h)', 'chunk','freq']) 
    
    # average the results of each chunk
    grouped_frame = mean_frame.groupby(['Quantity of speech (h)', 'freq']).mean()
    updated = grouped_frame.reset_index()
    hour_lst = []
    freq_lst = []
    k = 0
    while k < updated.shape[0]:
        hour_lst.append(int(updated['Quantity of speech (h)'].tolist()[k][:-1]))
        k += 1
    updated['Quantity of speech (h)'] = hour_lst  
    
    updated.to_csv(lang + '_Weighted_mean.csv')      



"""
Step2: x-axis unification
multiple averages: average on different words, different chunks(models)
"""
import matplotlib.pyplot as plt
import seaborn as sns

def plot_curves(lang):
    updated = pd.read_csv(lang + '_Weighted_mean.csv')
    freq_lst = []
    k = 0
    while k < updated.shape[0]:
        freq_lst.append(str(updated['freq'].tolist()[k]))
        k += 1
    updated['Frequency band'] = freq_lst 
    order = ['1', '2','4', '16','32', '64' ]
    sns.set_style('whitegrid')
    sns.lineplot(data=updated, x="Quantity of speech (h)", y="Lexical score", hue="Frequency band",hue_order=order)
    if lang == 'EN':
        fig_title = 'English model'
    elif lang == 'FR':
        fig_title = 'French model' 
            
    plt.title(fig_title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # Show the plot
    plt.show()
    plt.savefig(lang + '_model.png')


"""
Step3: compare curves
1. within model
the learning rate and freq in different quqanties of input

x: freq band
y:lr
split by different time intervals
counted as lexical scores every 10 hours

2.human-model comparison
under the same scale


def plot_slope():
    lang = 'EN'
    updated = pd.read_csv(lang + '_Weighted_mean.csv')
    
    #compare LR
    
    slope_lst =[]
    for band in list(set(updated['freq'].tolist())):
        temp = updated[updated['freq'] == band]
        sorted_group = temp.sort_values('Quantity of speech (h)')
        slope = sorted_group['Lexical score'].tolist()[-1] - sorted_group['Lexical score'].tolist()[0]
        slope_lst.append([band,slope])
    slope_frame = pd.DataFrame(slope_lst, columns=['freq', 'Lexical score change'])
    
    sns.set_style('whitegrid')
    sns.lineplot(data=slope_frame, x='freq', y="Lexical score change")
    if lang == 'EN':
        fig_title = 'English model'
    elif lang == 'FR':
        fig_title = 'French model' 
            
    plt.title(fig_title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # Show the plot
    plt.show()
    plt.savefig(lang + '_model.png')



def plot_procedural_slope():
    updated = pd.read_csv(lang + '_Weighted_mean.csv')
    updated['Lexical score'] = updated['Lexical score'] * 1000
    #compare LR
    df = pd.DataFrame()
    for band in list(set(updated['freq'].tolist())):
        slope_lst =[]
        interval_lst =[]
        temp = updated[updated['freq'] == band]
        sorted_group = temp.sort_values('Quantity of speech (h)')
        n = 0
        while n < sorted_group.shape[0]:
            if n == 0:
                slope = sorted_group['Lexical score'].tolist()[n]/sorted_group['Quantity of speech (h)'].tolist()[n] 
                interval = '0-' + str(sorted_group['Quantity of speech (h)'].tolist()[n])
            if n > 0:
                slope = (sorted_group['Lexical score'].tolist()[n] - sorted_group['Lexical score'].tolist()[n-1])/(sorted_group['Quantity of speech (h)'].tolist()[n] - sorted_group['Quantity of speech (h)'].tolist()[n-1])
                interval = str(sorted_group['Quantity of speech (h)'].tolist()[n - 1]) + '-' + str(sorted_group['Quantity of speech (h)'].tolist()[n])
            
            slope_lst.append(slope)
            interval_lst.append(interval)
            n += 1
        sorted_group['learing speed(lexical score increase per 10h)'] = slope_lst
        sorted_group['interval'] = interval_lst
        df = pd.concat([df, sorted_group]) 
    
    
    sns.set_style('whitegrid')
    sns.lineplot(data=df, x='freq', y="learing speed(lexical score increase per 10h)", hue="interval")
    if lang == 'EN':
        fig_title = 'English model'
    elif lang == 'FR':
        fig_title = 'French model' 
            
    plt.title(fig_title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # Show the plot
    plt.show()
    plt.savefig(lang + '_model.png')

"""



# get the detailed scores
def get_score(root_dir,lang):
    df = pd.DataFrame()
    mean_frame_lst = []
    # iterate all the folders
    for file in os.listdir(root_dir + lang):
        # iterate hours
        hour_name = os.fsdecode(file)
        try:
            for folder in os.listdir(root_dir + lang + '/' + hour_name):
                # iterate chunks
                chunk_name = os.fsdecode(folder)
                
                for subfoldername in os.listdir(root_dir + lang + '/' + hour_name + '/' + chunk_name +'/lstm/scores/swuggy/' + lang.lower()):
                    textfolder = os.fsdecode(subfoldername)
                    # read .csv file
                    result = pd.read_csv(root_dir + lang + '/' + hour_name + '/' + chunk_name +'/lstm/scores/swuggy/' + lang.lower() + '/' + textfolder + '/score_lexical_dev_by_pair.csv')           
                    
                    # use differeent thresholds to re-calculate the results
                    # threshold 1: 3/4 as known the word
                    score_lst_main = []
                    # threshold 2: 4/4 as known the word
                    score_lst_all = []
                    
                    
                    df = pd.concat([df, result])  
                    n = 0
                    while n < result.shape[0]:
                        
                        if result['score'].tolist()[n] >= 0.75:
                            score_main = 1
                        else:
                            score_main = 0
                        score_lst_main.append(score_main)
                        
                        if result['score'].tolist()[n] == 1:
                            score_all = 1
                        else:
                            score_all = 0
                        score_lst_all.append(score_all)
                        n += 1
                    mean_frame_lst.append([hour_name[:-1],chunk_name,textfolder.split('_')[-1],sum(score_lst_main)/len(score_lst_main),sum(score_lst_all)/len(score_lst_all)])
        except:
           print([folder,subfoldername]) 
   
    # get the means of the score
    mean_frame = pd.DataFrame(mean_frame_lst, columns=['Quantity of speech (h)', 'chunk','freq', 'Lexical score (3/4)','Lexical score (4/4)']) 
    # average the results of each chunk
    grouped_frame = mean_frame.groupby(['Quantity of speech (h)', 'freq']).mean()
    updated = grouped_frame.reset_index()
    updated.to_csv(lang + '_processed_score.csv')      
    return updated

'''
regroup the words based on medians
input: a dataframe ordered word freq as well as words
output:a new dataframe with added group index

return the chunked list
-> map this with the group index
'''

def chunk_list(input_list, num_groups, target_medians, n):
    # if input list is too short to be split into num_groups or n is greater than the length of the input list
    if len(input_list) < num_groups * n:
        return None
    
    # initialize list of subgroups with at least n elements each
    subgroups = [input_list[i:i+n] for i in range(0, len(input_list), n)]
    num_subgroups = len(subgroups)
    
    # if the number of subgroups is less than num_groups, pad the subgroups with the last element
    if num_subgroups < num_groups:
        last_subgroup = subgroups.pop()
        for i in range(num_groups - num_subgroups):
            subgroups.append(last_subgroup.copy())
    
    # calculate the median of each subgroup and sort them in ascending order
    subgroup_medians = [np.median(subgroup) for subgroup in subgroups]
    subgroup_medians.sort()
    
    # determine which target medians each subgroup should be assigned to
    assignments = []
    for subgroup_median in subgroup_medians:
        distances = [abs(subgroup_median - target_median) for target_median in target_medians]
        assignments.append(distances.index(min(distances)))
    
    # construct the final list of subgroups
    final_subgroups = [[] for _ in range(num_groups)]
    for subgroup, assignment in zip(subgroups, assignments):
        final_subgroups[assignment].extend(subgroup)
    
    # if any subgroup has less than n elements, try to merge it with the adjacent subgroups
    for i in range(num_groups):
        while len(final_subgroups[i]) < n and num_subgroups > num_groups:
            if i == 0:
                # merge with right neighbor
                final_subgroups[i].extend(final_subgroups[i+1])
                del final_subgroups[i+1]
                num_subgroups -= 1
            elif i == num_groups-1:
                # merge with left neighbor
                final_subgroups[i].extend(final_subgroups[i-1])
                del final_subgroups[i-1]
                num_subgroups -= 1
                i -= 1
            elif len(final_subgroups[i-1]) < len(final_subgroups[i+1]):
                # merge with left neighbor
                final_subgroups[i].extend(final_subgroups[i-1])
                del final_subgroups[i-1]
                num_subgroups -= 1
                i -= 1
            else:
                # merge with right neighbor
                final_subgroups[i].extend(final_subgroups[i+1])
                del final_subgroups[i+1]
                num_subgroups -= 1
        
    # if the final list of subgroups still has less than num_groups elements, return None
    if len(final_subgroups) < num_groups:
        return None
    
    return final_subgroups






def chunk_list(input_list, num_groups, target_medians):
    # calculate the median of the input list
    median = np.median(input_list)
    
    # split the list into two halves
    left_half = [x for x in input_list if x <= median]
    right_half = [x for x in input_list if x > median]
    
    # calculate the median of each half
    left_median = np.median(left_half)
    right_median = np.median(right_half)
    
    # if either half has fewer than half the elements of the original list, discard its median and try again
    if len(left_half) < len(input_list) / 2:
        return chunk_list(input_list, num_groups, target_medians)
    if len(right_half) < len(input_list) / 2:
        return chunk_list(input_list, num_groups, target_medians)
    
    # divide each half into two subgroups based on its median
    left_left = [x for x in left_half if x <= left_median]
    left_right = [x for x in left_half if x > left_median]
    right_left = [x for x in right_half if x <= right_median]
    right_right = [x for x in right_half if x > right_median]
    
    # recursively divide each subgroup until the desired number of groups is obtained
    subgroups = [left_left, left_right, right_left, right_right]
    while len(subgroups) < num_groups:
        new_subgroups = []
        for subgroup in subgroups:
            subgroup_median = np.median(subgroup)
            subgroup_left = [x for x in subgroup if x <= subgroup_median]
            subgroup_right = [x for x in subgroup if x > subgroup_median]
            if len(subgroup_left) >= len(subgroup) / 2 and len(subgroup_right) >= len(subgroup) / 2:
                new_subgroups.append(subgroup_left)
                new_subgroups.append(subgroup_right)
            else:
                # if a subgroup has fewer than half the elements of its parent subgroup, discard its median and try again
                new_subgroups.extend(chunk_list(subgroup, 2, [np.median(subgroup)]))
        subgroups = new_subgroups
    
    # calculate the median of each subgroup and sort them in ascending order
    subgroup_medians = [np.median(subgroup) for subgroup in subgroups]
    subgroup_medians.sort()
    
    # determine which target medians each subgroup should be assigned to
    assignments = []
    for subgroup_median in subgroup_medians:
        distances = [abs(subgroup_median - target_median) for target_median in target_medians]
        assignments.append(distances.index(min(distances)))
    
    # construct the final list of subgroups
    final_subgroups = [[] for _ in range(num_groups)]
    for i, subgroup in enumerate(subgroups):
        final_subgroups[assignments[i]].extend(subgroup)
    
    return final_subgroups

'''
output two types of lexical scores
with the month as x-axis
'''
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
    
    return final_index_lst


def plot_curves(lang,lexical_socre):
    updated = pd.read_csv(lang + '_processed_score.csv')
    freq_lst = []
    k = 0
    while k < updated.shape[0]:
        freq_lst.append(str(updated['freq'].tolist()[k]))
        k += 1
    updated['Frequency band'] = freq_lst 
    order = ['4', '16','32', '64' ]
    sns.set_style('whitegrid')
    sns.lineplot(data=updated, x="month", y=lexical_socre, hue="Frequency band",hue_order=order)
    plt.xlim(0,36)
    plt.ylim(0,1)
    if lang == 'EN':
        fig_title = 'English model'
    elif lang == 'FR':
        fig_title = 'French model' 
            
    plt.title(fig_title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # Show the plot
    plt.show()
    plt.savefig(lang + 'normalized_model.png')


plot_curves('FR','Lexical score (4/4)')


'''
get freq statistics

1. get the raw freq
aggregate results of the whole corpous
or count by your self

2.get the normalized freq


3. concatenate the results altogether in a whole dataframesummary.columns = [('frequency', 'Count'), ('frequency', 'Median'), ('frequency', '5th Percentile'), ('frequency', '95th Percentile')]summary.columns = [('frequency', 'Count'), ('frequency', 'Median'), ('frequency', '5th Percentile'), ('frequency', '95th Percentile')]


Take the infant data as reference
'''

def get_statistics(lang, infant_file):
    # REad the refernce data
    reference  = pd.read_csv()
    # get the merged dataframe
    folder = 'STELA/Freq/' + lang + '/'
    df = pd.DataFrame()
    for file in os.listdir(folder):
        raw_freq = pd.read_csv(folder + file).dropna()
        # remove duplicated values
        clean_freq = raw_freq.drop_duplicates(subset=['word'])
        # get the normalized freq
        
        normalized_freq = clean_freq['frequency']/28800000
        clean_freq['frequency'] =  normalized_freq * 1000000
        # group list based on infants data
        
        # get the updated dataframe
        
        clean_freq['Group'] = int(file.split('.')[0].split('_')[-1])
        # Group by 'Group' column and calculate length, median, 5th and 95th percentile
        summary = clean_freq.groupby('Group').agg({'frequency': ['count', 'median', lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)]})
        
        # Rename columns
        
        summary.columns = [('frequency', 'Count'),('frequency', '5th Percentile'), ('frequency', 'Median'),  ('frequency', '95th Percentile')]
        # Reset index and rename columns
        summary = summary.reset_index()
        summary.columns = ['Group', 'Count', '5th Percentile', 'Median', '95th Percentile']
        
        df = pd.concat([df, summary])  
        
    # convert into int format
    group_lst = []
    for num in df['Group'].tolist():
        group_lst.append(int(num))
    
    df['Group'] = group_lst 
    sorted_summary = df.sort_values('Group')
    sorted_summary.to_csv('STELA/' + lang + '_stat.csv')
 
    
lang = 'EN'
infant_file = 'Infants'
reference  = pd.read_csv('STELA/' + infant_file + '.csv')

# replace 'AE' or 'BE' with 'EN'
reference['Lang'] = reference['Lang'].replace(['AE', 'BE'], 'EN')
# group the dataframe based on language and group
aggregated_ref = reference.groupby(['Group','Lang'])['0.5'].mean().reset_index()

# get the merged dataframe
folder = 'STELA/Freq/' + lang + '/'
df = pd.DataFrame()
for file in os.listdir(folder):
    raw_freq = pd.read_csv(folder + file).dropna()
    
    # remove duplicated values
    # get another list of scores
    clean_freq = raw_freq.drop_duplicates(subset=['word'])
    temp = reference.groupby(['word'])['0.5'].mean().reset_index()
    
    # get the normalized freq
    normalized_freq = clean_freq['frequency']/28800000
    clean_freq['Norm_freq_per_million'] =  normalized_freq * 1000000
    
    df = pd.concat([df, clean_freq])  


    
# get logarithm of normalized word freq per million
norm_log_freq_lst = []
for freq in df['Norm_freq_per_million'].tolist():
    norm_log_freq = math.log2(freq)
    norm_log_freq_lst.append(norm_log_freq)
df['Log_norm_freq_per_million'] = norm_log_freq_lst
# sort the concatenated results by freq
clean_freq = df.sort_values('Norm_freq_per_million')

# for the given language
target_medians = aggregated_ref.loc[aggregated_ref['Lang'] == lang]['0.5'].tolist()

# group list based on infants data, we ensure there are at least 100 vlaues for each subgroup
chunked_lst = chunk_list(clean_freq['Norm_freq_per_million'].tolist(), len(target_medians), target_medians)
# convert into group index
# output the corresponding indices
final_index_temp = []
fre_band = []
group_num = 4
for num in range(group_num):
    fre_band.append(str(2**num))
    
for i in range(group_num):  
    final_index_temp.append([fre_band[i]] * len(chunked_lst[i]))
    

# flatten the list
final_index_lst = [item for sublist in final_index_temp for item in sublist]      
  
clean_freq['Group'] = final_index_lst  


  
# Group by 'Group' column and calculate length, median, 5th and 95th percentile
summary = clean_freq.groupby('Group').agg({'frequency': ['count', 'median', lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)]})
    
# Rename columns
    
summary.columns = [('frequency', 'Count'),('frequency', '5th Percentile'), ('frequency', 'Median'),  ('frequency', '95th Percentile')]
    # Reset index and rename columns
summary = summary.reset_index()
summary.columns = ['Group', 'Count', '0.05', '0.5', '0.95']
    
    
# convert into int format
group_lst = []
for num in df['Group'].tolist():
    group_lst.append(int(num))

df['Group'] = group_lst 
sorted_summary = df.sort_values('Group')
sorted_summary.to_csv('STELA/' + lang + '_stat.csv')

def get_statistics(lang):
    
    folder = 'STELA/Freq/' + lang + '/'
    df = pd.DataFrame()
    for file in os.listdir(folder):
        raw_freq = pd.read_csv(folder + file).dropna()
        # remove duplicated values
        clean_freq = raw_freq.drop_duplicates(subset=['word'])
        # get the normalized freq
        if lang == 'EN': 
            normalized_freq = clean_freq['frequency']/28800000
        clean_freq['frequency'] =  normalized_freq * 1000000
        # get the concatenated list and then get the stat by group
        clean_freq['Group'] = int(file.split('.')[0].split('_')[-1])
        # Group by 'Group' column and calculate length, median, 5th and 95th percentile
        summary = clean_freq.groupby('Group').agg({'frequency': ['count', 'median', lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)]})
        
        # Rename columns
        #summary.columns = [('frequency', 'Count'), ('frequency', 'Median'), ('frequency', '5th Percentile'), ('frequency', '95th Percentile')]
        summary.columns = [('frequency', 'Count'),('frequency', '5th Percentile'), ('frequency', 'Median'),  ('frequency', '95th Percentile')]
        # Reset index and rename columns
        summary = summary.reset_index()
        summary.columns = ['Group', 'Count', '5th Percentile', 'Median', '95th Percentile']
        
        df = pd.concat([df, summary])  
        
    # convert into int format
    group_lst = []
    for num in df['Group'].tolist():
        group_lst.append(int(num))
    
    df['Group'] = group_lst 
    sorted_summary = df.sort_values('Group')
    sorted_summary.to_csv('STELA/' + lang + '_stat.csv')
    
get_statistics('FR')   

# numn_words = 459,574,310  28,800,000

lang = 'EN'
folder = 'STELA/Text/' + lang + '/' 
num_words = 0
# merge to all
# open the file and read its contents
for filename in os.listdir(folder):
    with open(folder + filename, 'r', encoding='UTF-8') as file:
        data = file.read()
        # Split the string into lines
        lines = data.split("\n")
        
        # Remove empty lines
        non_empty_lines = [line for line in lines if line.strip() != ""]
        
        # Join the non-empty lines back into a string
        result = "\n".join(non_empty_lines)
        
        # remove punctuations
        translator = str.maketrans('', '', string.punctuation + string.digits)
        clean_string = result.translate(translator)
        words = clean_string.split()
    
        # Count the number of words
        num_words += len(words)
        
    
    
    
'''
regroup words into different chunks based on freq bands



write python codes to chunk a list ordered numbers so that each subgroups' medians are as close as possible to the given number list. It is allowed to discard numbers to achieve the closest median. But the proportion of discarded numbers should be lower than 0.5 

'''

    
    
    
    
    
    