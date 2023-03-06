# -*- coding: utf-8 -*-
"""
Plot loss to check hyperparameters 
"""
import matplotlib.pyplot as plt

# Set the path to the fairseq log file
log_file = 'result.log'

# Parse the log file and extract the train loss values
raw = []
val_index_lst = []
train_index_lst = []
with open(log_file, 'r') as f: 
    n = 0
    for line in f: 
        raw.append(line)
        if 'end of epoch' in line:     
            val_index_lst.append(n + 1)
        if 'begin validation on "valid" subset' in line:     
            train_index_lst.append(n - 1)    
        n += 1

val_loss_lst = [] 
train_loss_lst = [] 
for index in val_index_lst:
    val_loss = float(raw[index].split('| ')[1].split(' ')[1])
    val_loss_lst.append(val_loss)   

for index in train_index_lst:
    train_loss = float(raw[index].split('=')[1].split(',')[0])
    train_loss_lst.append(train_loss)  
    

# create data
x = []
for n in range(len(val_loss_lst)):
    x.append(n)
# plot lines
plt.plot(x, val_loss_lst, label='Val')
plt.plot(x, train_loss_lst, label='Train')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss history")
plt.legend()
plt.show()

