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
Step 1: ASR on subaudios using whisper

input: tree_path, metadata_path, audio_directory, language
    
    1. tree_path  e.g."/scratch1/projects/InfTrain/dataset/trees"
    2. metadata_path  e.g. '/scratch1/projects/InfTrain/dataset/metadata/matched2.csv'
    3. audio_directory e.g.'/scratch1/projects/InfTrain/dataset/wav
    4. language  e.g.'FR', "EN"

output: 
    1.subaudio transcriptions with the same name as the subaudio files in a folder named 'EN' or 'FR'
    2.log.csv file for all the errors occurring in ASR process
    3.path.csv file with the selcted audio files
'''

# we use the multilingual ASR model
asr_model = whisper.load_model("base")

# support func for sorting
'''
# this one is originally used for ASR part, possible to be integrated below
def natural_keys(text):
    return [text.split('.')[-2].split('_')[-1]] 
'''

def natural_keys(text):
    return [text.split('.')[0].split('_')[-1]] 


def transcribe_audio(tree_path, metadata_path, audio_directory, lang):
    '''
    step 1: get audio paths
    step 2: transcribe audios and save them in a folder
    
    different languages: different subaudio list; name suffix for the chosen file; asr model setting
    
    '''
    log_lst = []
    metadata = pd.read_csv(metadata_path)
    
    # select the audio names from the json tree data
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
            audio_path = audio_directory + '/' + lang + '/' + metadata[metadata.book_id == audio]['speaker_id'].item().split(',')[0].split('\'')[1] + '/' + audio
            
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


'''
Step 2: match the automatically recognized texts with textual data

input: metadata_path, audio_path. text_path
    For instance: 
    tree_path = "/scratch1/projects/InfTrain/dataset/trees"
    metadata_path = '/scratch1/projects/InfTrain/dataset/metadata/matched2.csv'
    language = 'FR'

output: subaudio transcriptions with the same name as the subaudio files

'''


# audio_path = './data/Sample_audio/1121/1812_LibriVox_en'
# text_path = './data/Text/LibriVox/adventures_jimmy_skunk_jl_librivox_64kb_mp3_text.txt'

def get_text(tree_path, metadata, language, audio_list):
    
    # get the text list
    text_lst = []
    for audio in audio_list:
        text_lst.append(metadata[metadata.book_id == audio]['text_path'].item().split('/')[-1])
    return text_lst


'''
# not possible to base on only one word
# rather, use 3-gram to locate the first matching part
# get the first matching location and remove the strings before that
lower case for mamtching: this is plausible as it won't change the position of th certain char
punc would possibly be a problem: only preserve ,.?!/'
-> for /'. do I need a special token of this? 
-> try different conditions: 1.with punct; 2.withut punct; 3.with influential punct
'''



def matching_part(text, trans, section):
    text_lowered = text.lower()
    trans_lowered = trans.lower()

    trans_lst = trans_lowered.split(" ")
    key_lst = []
    n = 0
    while n < len(trans_lst):
        string = ''
        for word in trans_lst[n:n+5]:
            string +=  word + ' '
        key_lst.append(string[:-1])
        
        n += 1
    text_index_lst = []
    trans_index_lst = []
    
    for key in key_lst[:-4]:
            # if there is matching part
            if text_lowered.find(key) != -1:
                text_index_lst.append(text_lowered.find(key))
                trans_index_lst.append(trans_lowered.find(key))
                
    if section == "begin":
        
        if len(trans_index_lst) > 0:
            matched = trans[:trans_index_lst[0]] + text[text_index_lst[0]:]
        else:
            matched = trans + text
    
    else:
        
        if len(trans_index_lst) > 0:
            # attach them together: use the overlapping part as the anchoring 
            matched = text[:text_index_lst[-1]] + trans[trans_index_lst[-1]:]
        else:
            matched = text + trans 
        
    return matched

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
return one cleaned file
input: audio file name; text file name
output: the cleaned text file
'''

def clean_trans(audio, trans_path, text_path, subaudio_directory,metadata,language):
  # match the audio file with the corresponding subaudios
  subaudio_lst = []
  log_lst = []
  for i in subaudio_directory:
      
      try:
          audio_temp = i.split('.')[0].split('_')[:3]
          file = audio_temp[0] +'_'+ audio_temp[1] + '_'+ audio_temp[2]
          if file == audio:
              subaudio_lst.append(i)
              # sort the subaudios based on the sequence number
      except:
          if i == "EN_Log.txt" or 'EN_log.csv' or 'EN_audioPath.csv':
              pass
          else:
             log_lst.append(i) 
             
  subaudio_lst.sort(key=natural_keys)
  # get the script list
  script_lst = []
  # trick: in the case of redundent transcriptions in En folders, taking the first and last two files would be safest choice
  for subaudio in [subaudio_lst[0],subaudio_lst[1],subaudio_lst[-2],subaudio_lst[-1]]: 
      #perform ASR on all the audio data
      with open(trans_path + '/' + subaudio, encoding='UTF-8') as file:
          script = file.read()
          script_lst.append(script)


  # open the corresponding text file
  text_file = metadata[metadata.book_id == audio]['text_path'].item().split('/')[-1]
  text = text_path + '/' + text_file
  with open(text, encoding='UTF-8') as f:
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
      begin_trans = script_lst[0] + script_lst[1]
      end_trans = script_lst[-2] + script_lst[-1]
      # second audio: used for match
      overlap_start = matching_part(begin, begin_trans,"begin")
      overlap_end = matching_part(end, end_trans,"end")
      
      
      matched = overlap_start + overlap_end
      
      # save the file with the same name
      file = open('./cleaned/'+ language + '/' + text_file,"w+")
      file.write(matched) 
  
  return log_lst

'''
metadata_path = './Data/metadata/matched2.csv'        
metadata = pd.read_csv(metadata_path)
trans_path = './example'
text_path = './Data/Text/LibriVox'

audio = '4220_LibriVox_en'
language = "EN"
# get a list of subaudio filenames
subaudio_directory = []    
for file in os.listdir(trans_path):
      filename = os.fsdecode(file)
      subaudio_directory.append(filename)

clean_trans(audio, trans_path, text_path, subaudio_directory,metadata,language)
'''


metadata_path = '/scratch1/projects/InfTrain/dataset/metadata/matched2.csv'        
metadata = pd.read_csv(metadata_path)
trans_path = '/scratch2/jliu/STELAWord/ASR/EN'
text_path = '/scratch1/projects/InfTrain/dataset/text/EN/LibriVox'
language = "EN"
# get a list of subaudio filenames
subaudio_directory = []    
for file in os.listdir(trans_path):
      filename = os.fsdecode(file)
      subaudio_directory.append(filename)
      
audio_lst = pd.read_csv('/scratch2/jliu/STELAWord/ASR/EN/EN_audioPath.csv',header = None)[0].tolist()
log_lst = []
for path in audio_lst:    
    audio = path.split('/')[-1]
    log = clean_trans(audio, trans_path, text_path, subaudio_directory,metadata,language)
    if len(log) > 0:
        log_lst.append(log)
        
log_csv = pd.DataFrame(log_lst)    
log_csv.to_csv('./cleaned/'+ language + '/' + language + "_Log.csv", index=False, header=False) 

'''

match audio with text name
input: tree_path, metadata, audiopath
output: pre-processed datafile and two jason files in correspondence with the audio files

'''



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














