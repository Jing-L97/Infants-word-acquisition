#############################################################
# simulation of lexical learning under an accumulator model #
# for an artificial ngram model under zipf law              #
#############################################################

library(Matrix)
library(lme4)
library(lmerTest)
library(emmeans)
# word "activation" function
# input: array of word counts
# output: array of word "activation" levels (between 0:unknown and 1:known)
# here, just a linear function truncated at 10 counts.
wactiv<-function(count){
  ifelse(count>10,1,count/10)
}

data <- read.csv("C:/Users/Crystal/OneDrive/Desktop/Trial.csv")

### creating a frequency table for a lexicon of size K ('words' are just the indices 1..K)
# tokenize and pre-process our corpus to get the vocab size
K= 200 # vocabulary size (number of words)
zipf_freq = data$Freq
# here: the word freq can be counted instead of using zipf law
#zipf_freq=trunc(1/(1:K)*K*10)  # frequency count of each word (according to zipf law)
#plot(zipf_freq)
print(zipf_freq)

### creating a corpus (unigramly according to the frequency table) of size N
corpus=NULL # text corpus
for(i in 1:K){
  corpus=c(corpus,rep(i,zipf_freq[i]))
}
N=length(corpus) # size of the corpus


#corpus=NULL # text corpus
#for(i in 1:K){
  #corpus=c(corpus,rep(i,zipf_freq[i]))
  #}
#N=length(corpus) # size of the corpus

### computing cumulative counts and activity as a function of input data size 
meanacti=matrix(0,K,N) # holds the mean activation of each word, as a function of time (corpus size)

## computing the average activation level for each word as a function of amount of data
NREP=200  # number of resampling in the corpus
for (j in 1:NREP){
  activ=matrix(0,K,N) # contains activation of each word as a function of corpus size
  x=sample(corpus) # shuffling the corpus (size N)
  for (k in 1:K)
     activ[k,]=wactiv(cumsum(x==k)) # computing the learning curve for each word

  meanacti=meanacti+activ/NREP # averaging this across the NREP resampling
}

## shows the activation plots for a representative sample of words (equally spaced in log frequency) 
plot(1:N,meanacti[1,],type='l',xlim=c(1,N),xlab='Time (nb of words)',ylab='Percent learned')
for(q in 1:(log(K)/log(2))){
  lines(1:N,meanacti[2**q,])
}

## same in log space
plot(1:N,meanacti[1,],type='l',xlim=c(1,N),log='x',xlab='Time (nb of words)',ylab='Percent learned')
for(q in 1:(log(K)/log(2))){
  lines(1:N,meanacti[2**q,])
}

