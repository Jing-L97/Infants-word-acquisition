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
import json
import whisper

'''
load whisper asr model
'''
asr_model = whisper.load_model("base")


'''
match audio with text name
input: tree_path, metadata, audiopath
output: pre-processed datafile and two jason files in correspondence with the audio files
'''

audio_path = './data/Sample_audio/1121/1812_LibriVox_en'
text_path = './data/Text/LibriVox/adventures_jimmy_skunk_jl_librivox_64kb_mp3_text.txt'

def natural_keys(text):
    return [text.split('.')[-2].split('_')[-1]] 

def transcribe_audio(tree_path, metadata_path,lang):
    '''
    step 1: get audio paths
    step 2: transcribe audios and save them in a folder
    
    different languages: different sudio list; name suffix for the chosen file; asr model setting
    
    '''
    log_lst = []
    metadata = pd.read_csv(metadata_path)
    #get the audion file name 
    
    
    if lang == 'EN':
        audio_suffix = r"\d+_LibriVox_en" 
        selected_lang = 'English'
        tree = tree_path + '/' + lang + '.json'
        
    elif lang == 'FR':
        audio_suffix = r"\d+_LibriVox_fr|\d+_LitteratureAudio_fr" 
        selected_lang = 'French'
        tree = tree_path + '/' + lang + '.json'
        
    # different tree files for different languages
    with open(tree, 'r') as f:
      data = json.load(f)
      
     # save transcriptions in the corresonding directories; make one if there's no
    output_dir = "./" + lang 
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)
            
        # get audio file list
    audio_list = re.findall(audio_suffix,str(data))
        # remove duplicate
    audio_list_clean = list(set(audio_list)) 
        
    audio_path_lst = []
    for audio in audio_list_clean:
            audio_path = '/scratch1/projects/InfTrain/dataset/wav/' + lang + '/' + metadata[metadata.book_id == audio]['speaker_id'].item().split(',')[0].split('\'')[1] + '/' + audio
            
            audio_path_lst.append(audio_path)
            
            # iterate the subfolders and only do ASR on the first and last 2 audios
            subaudio_lst = []    
            for file in os.listdir(audio_path):
                filename = os.fsdecode(file)
                subaudio_lst.append(filename)
            
            # sort the audios so that it will ensure to get the first and last two
            subaudio_lst.sort(key=natural_keys)
            
            
            try: 
                
                for subaudio in [subaudio_lst[0],subaudio_lst[1],subaudio_lst[-2],subaudio_lst[-1]]: 
                    result = asr_model.transcribe(audio_path + '/' + subaudio,fp16=False, language=selected_lang)
                    
                    # change the suffix by removing .wav
                    f = open(output_dir + '/'+ subaudio[:-4] + ".txt","w+")
                    f.write(result["text"])
                    
            
            except:
                log_lst.append(audio)
        
    # print out the history   
    log_csv = pd.DataFrame(log_lst)
    log_csv.to_csv(output_dir + '/'+ lang + "_log.csv", index=False, header=False)
    
    path_csv = pd.DataFrame(audio_path_lst)
    
    path_csv.to_csv(output_dir + '/'+ lang + "_audioPath.csv", index=False, header=False)
     
    return audio_path_lst





def get_text(tree_path, metadata, language):
    #get the audion file name 
    with open(tree_path, 'r') as f:
      data = json.load(f)
    
    if language == 'EN':
        audio_list = re.findall(r"\d+_LibriVox_en",str(data))
    
    # save as the corresponding text files in the tree structure; useful for training !!!
    
    # remove duplicate audio files
    audio_list_clean = list(set(audio_list)) 
    
    # find corresponding text files
    text_lst = []
    for audio in audio_list_clean:
        text_lst.append(metadata[metadata.book_id == audio]['text_path'].item().split('/')[-1])
    return text_lst

tree_path = './data/Trees/EN.json'
metadata_path = './data/metadata/matched2.csv'
language = 'EN'
# read meta-data
metadata = pd.read_csv(metadata_path)
text_lst = get_text(tree_path, metadata, language)

'''
# not possible to base on only one word
# rather, use 3-gram to locate the first matching part
# get the first matching location and remove the strings before that
lower case for mamtching: this is plausible as it won't change the position of th certain char
punc would possibly be a problem: only preserve ,.?!/'
-> for /'. do I need a special token of nthis? 
-> try different conditions: 1.with punct; 2.withut punct; 3.with influential punct
'''

def matching_part(string1, string2):
    m = len(string1)
    n = len(string2)
    
    string1 = string1.lower()
    string2 = string1.lower()
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
            # calulate the space to get # words  
            # possible problem: here we also take the punctuation into account
        if result.count(" ") >= 3:
            matched_lst.append(result)        
        else:
            pass  
    return matched_lst

'''
1st edition: remove all punct
    reserved for model phonemization
'''


def remove_punct(line: str) -> str:
    """This function will remove any non-alphabetic characters from any of the texts"""
    non_letters_re = re.compile(r"[@!'--\"%(),€./#:;&=?\[\]\^_<>$«°»\n\\]")
    return non_letters_re.sub(" ", line)

'''
2nd edition: reserve punct supporting segmentations(!.,?:')
    pay attention to the symbol \', might be a problem
'''

def remove_format(line: str) -> str:
    """This function will remove any non-alphabetic characters from any of the texts"""
    non_letters_re = re.compile(r"[-%()€/#&=\[\]\^_<>$«°»\n\\]")
    return non_letters_re.sub(" ", line)

'''
Attention: here 
'''

def clean_trans(audio_path, text_path):
  '''
  subaudio_lst = []    
  for file in os.listdir(audio_path):
      filename = os.fsdecode(file)
      subaudio_lst.append(filename)
  '''
  subaudio_lst = ['1812_LibriVox_en_seq_00.wav','1812_LibriVox_en_seq_01.wav','1812_LibriVox_en_seq_68.wav','1812_LibriVox_en_seq_69.wav']    

  # only work on the first two and the last one
  script_lst = []
  for subaudio in [subaudio_lst[0],subaudio_lst[1],subaudio_lst[-2],subaudio_lst[-1]]: 
      #perform ASR on all the audio data
      result = eng_asr_model.transcribe(subaudio)
      script = result["text"]
      script_lst.append(script.lower())

  with open(text_path, encoding='UTF-8') as f:
      # reading each line 
      lines = f.read()
      cleaned_word_temp = remove_format(lines) 
      
      # set the text proportion as twice of the audio to ensure covering the actual content   
      begin = cleaned_word_temp[:int(len(cleaned_word_temp) * 0.5)]
      end = cleaned_word_temp[int(len(cleaned_word_temp) * 0.5):]

      '''
      list structure: [opening, 1st ele, 2nd ele, ending]
      1st and 2nd ele: decide the start and end parts of the text parts
      split into 2 and concatenate them
      opening and ending:added as extra
      '''
      # second audio: used for match
      overlap_start = matching_part(begin, script_lst[1])[0]    
      # as the outputs are in the lower case, check this regardless of the case as well
      start_index = begin.lower().find(overlap_start)
      # remove the unnecessary part in the end
      overlap_end = matching_part(end, script_lst[2])[-1] 
      end_index = end.find(overlap_end)
      matched = script_lst[0] + begin[start_index:] + end[:end_index] + script_lst[-1]
      # save the file with the same name
      
  return matched

audio_path = './data/Sample_audio/1121/1812_LibriVox_en'
text_path = 'adventures_jimmy_skunk_jl_librivox_64kb_mp3_text.txt'

matched = clean_trans(audio_path, text_path)

'''
this is only for one text 
'''
def preprocess_text(tree_path, metadata, language):
    text_lst = get_text(tree_path, metadata, language)
    

'''

match audio with text name
input: tree_path, metadata, audiopath
output: pre-processed datafile and two jason files in correspondence with the audio files

'''


def main():
    preprocess_text(tree_path, metadata, language)
    
if __name__ == "__main__":
    main()

'''
Q: additional processing steps for BPE model?
  word-based model only tokenization
'''
#####################
# word-preprocessing#
#####################

'''
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



additional step: check word distributions between wordbank and librivox
-> plot the freq curve and compare the results
how comparable they are
'''

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














