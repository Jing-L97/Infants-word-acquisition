# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 14:46:56 2022
This is for the data-preprocessing as the input of the Accumulator model
We consider two conditions: with and without morphological change 

@author: Crystal
"""

import os
import re
import pandas as pd
import collections
import json
import whisper

model = whisper.load_model("base")

'''
# match audio with text#
Current problem: higher requirement to run locally(speechbrain: incomplete result)
alternative though croarse: use text pre-processing first and proceed with the language modeling
'''

audio_path = './data/Sample_audio/1121/1812_LibriVox_en'
text_path = './data/Text/LibriVox/adventures_jimmy_skunk_jl_librivox_64kb_mp3_text.txt'

#get the audion file name 
with open('./data/Trees/EN.json', 'r') as f:
  data = json.load(f)

audio_list = re.findall(r"\d+_LibriVox_en",str(data))

# read meta-data
meta = pd.read_csv('./data/metadata/matched2.csv')

# save as the corresponding text files in the tree structure

# remove duplicate audio files
audio_list_clean = list(set(audio_list)) 

# find corresponding text files
text_lst = []
for audio in audio_list_clean:
    text_lst.append(meta[meta.book_id == audio]['text_path'].item().split('/')[-1])


'''
match the asr result with the text transcription
fuzzy match: levenshtein distance is smaller than the threshold
beginning: first 2 audios(add the opening ones)
end: delete the unnecessary info

input: audio and raw text
output: cleaned text
'''
# enter the target folder
# perform the ASR 
subaudio_lst = []    
for file in os.listdir(audio_path):
    filename = os.fsdecode(file)
    subaudio_lst.append(filename)
# only work on the first two and the last one
script_lst = []
for subaudio in [subaudio_lst[0],subaudio_lst[1],subaudio_lst[-1]]: 
    #perform ASR on all the audio data
    script = eng_asr_model.transcribe(audio_path + '/'+ subaudio)
    script_lst.append(script.lower())
subaudio_lst[-1]
   
model.transcribe('1812_LibriVox_en_seq_00.wav')
# beginning
# the search scope: first part of the doc(divided by # sub-audios)
def clean_line(line: str) -> str:
    """This function will remove any non-alphabetic characters from any of the texts"""
    non_letters_re = re.compile(r"[@!'--\"%(),€./#0123456789:;&=?\[\]\^_<>$«°»\n\\]")
    return non_letters_re.sub(" ", line)

with open(text_path, encoding='UTF-8') as f:
    # reading each line 
    lines = f.read().lower()
    cleaned_word_temp = clean_line(lines)  
    proportion = 1/len(subaudio_lst)
    # set the text proportion as twice of the audio to ensure covering the actual content
    length = int(len(cleaned_word_temp) * proportion)
    begin = cleaned_word_temp[0:length*4]
    end = cleaned_word_temp[-length*2:]

# not possible to base on only one word
# rather, use 3-gram to locate the first matching part
# get the first matching location and remove the strings before that
def matching_part(string1, string2):
    m = len(string1)
    n = len(string2)
    result = ""
    
    dp = [[0 for x in range(n + 1)] for y in range(m + 1)]
    matched_lst = []
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp[i][j] = 0
            elif string1[i - 1] == string2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1              
                result = string1[i - dp[i][j]: i]
            else:
                dp[i][j] = 0
            #calulate the space to get # words        
        if result.count(" ") >= 3:
            matched_lst.append(result)        
        else:
            pass  
    return matched_lst


# second audio: used for match
# get the first element of the matching part  
overlap_start = matching_part(begin, script_lst[1])[0]    
start_index = begin.find(overlap_start)
# first audio:added as extra
matched_temp = script_lst[0] + cleaned_word_temp[start_index:]  
# remove the unnecessary part in the end
overlap_end = matching_part(end, script_lst[2])
end_index = end.find(overlap_end)
matched_end = script_lst[0] + end[:end_index]



#####################
# word-preprocessing#
#####################
# read the text transcriptions
text_EN_dir = '/scratch1/projects/InfTrain/dataset/'
text_EN = []
folder = os.fsencode(text_EN_dir)
output_dir = './data/cleaned_texts'
if not(os.path.exists(output_dir)):
    os.mkdir(output_dir)
    

raw_text_temp = []    
for file in os.listdir(folder):
    filename = os.fsdecode(file)
    # create a cleaned file folder 
    with open(text_EN_dir + filename, encoding='UTF-8') as f:
        # reading each line 
        lines = f.read()
        # remove numbers and punctu
        cleaned_word_temp = clean_line(lines) 
        # output the cleaned text
        cleaned_word = cleaned_word_temp.split()
        raw_text_temp.append(cleaned_word)
    
# flatten the list
raw_text = [word.lower() for sublist in raw_text_temp for word in sublist]
frequencyDict = collections.Counter(raw_text)  
freq_lst = list(frequencyDict.values())
word_lst = list(frequencyDict.keys())

# get freq
fre_table = pd.DataFrame([word_lst,freq_lst]).T
col_Names=["Word", "Freq"]
fre_table.columns = col_Names
fre_table.to_csv('Freq.csv')


# condition 2:with the acquisition of morphological rules
# tokenize all the words



#############################
# Get phonetic transcription#
#############################
'''
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
word = "Bonjour"
backend = EspeakBackend('fr-fr', language_switch="remove-utterance")
separator = Separator(word=None, phone=" ")
phonemized = backend.phonemize([word], separator=separator)[0].strip().split(" ")
phonemized

def phonememize(word):
  backend = EspeakBackend('fr-fr', language_switch="remove-utterance")
  separator = Separator(word=None, phone=" ")
  phonemized = backend.phonemize([word], separator=separator)[0].strip().split(" ")
  return phonemized

#use filtered word list in step 1
filtered_words_generated = pd.read_csv('Filtered_words_created.csv')['Word'].tolist()
precounted_words = pd.read_csv('./data/filtered_words.csv',header = None,delimiter = ' ')
#filter the precounted words
filtered_words = set(filtered_words_generated).intersection(precounted_words[0].tolist())
#match the filtered words with counts
filtered_words_df = precounted_words.loc[precounted_words[0].isin(filtered_words_generated)]
filtered_words_df.to_csv('filtered_phoneme.csv')

# filter those composed of more than 2 phonemes
pronun = []
phoneme_no = []
for word in filtered_words_df[0].tolist():
  phonemized_word = phonememize(word)
  pronun.append(phonemized_word)
  phoneme_no.append(len(phonemized_word)) 
headers =  ["Word", "Count"]
filtered_words_df.columns = headers
filtered_words_df['Pronun'] = pronun
filtered_words_df['Phoneme_no'] = phoneme_no
filtered_words_df.to_csv('Phonemized_backup.csv')
filtered_words_final = filtered_words_df[filtered_words_df['Phoneme_no'] > 2] 
filtered_words_final.to_csv('Phonemized_word.csv')   
'''














