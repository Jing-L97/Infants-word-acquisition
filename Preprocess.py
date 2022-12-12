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
    

def clean_line(line: str) -> str:
    """This function will remove any non-alphabetic characters from any of the texts"""
    non_letters_re = re.compile(r"[@!'--\"%(),€./#0123456789:;&=?\[\]\^_<>$«°»\n\\]")
    return non_letters_re.sub(" ", line)

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














