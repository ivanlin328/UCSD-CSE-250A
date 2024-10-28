import numpy as np
import matplotlib.pyplot as plt
##read data from txt.file
def load_file(file_voc,file_uni,file_big):
    with open(file_voc, 'r') as file:
        vocb= file.readlines()   
    with open(file_uni, 'r') as file:
        unigram= file.readlines()
    with open(file_big, 'r') as file:
        bigram= file.readlines()
    return vocb,unigram,bigram
    
## P_u(w) from letters start with M or m
def unigram_M(vocb,unigram):
    vocbs=[v.strip() for v in vocb]
    unigrams=np.array([int(u.strip()) for u in unigram])
    prob_u={}
    
    for i in range(len(vocb)):
        prob_u[vocbs[i]]=unigrams[i]/np.sum(unigrams)
        if(vocbs[i][0]=="M" or vocbs[i][0]=="m"):
            print(vocbs[i]+ " " +str(prob_u[vocbs[i]]))
    return prob_u

## Print out the most likely words to follow the word "The" 
import numpy as np

def bigram_The(bigram):
    prob_b = {}
    the_bigrams = {}
    partial_sums = {}
    vocbs = [v.strip() for v in vocb]  
    bigrams = [b.strip().split('\t') for b in bigram]
    bigrams = np.array(bigrams).astype(np.uint32)  
      
    for b in bigrams:
        index1 = int(b[0])
        index2 = int(b[1])
        if index1 not in partial_sums:
            partial_sums[index1] = np.sum(bigrams[bigrams[:, 0] == index1, 2])
        key = (vocbs[index1 - 1], vocbs[index2 - 1])
        value = int(b[2])
        prob_b[key] = value / partial_sums[index1]
    
   
    the_bigrams = {k: v for k, v in prob_b.items() if k[0] == "THE"}
    sorted_the_bigrams = sorted(the_bigrams.items(), key=lambda item: item[1], reverse=True)
    for bigram, prob in sorted_the_bigrams[:10]:
        print(f"{bigram}: {prob}")
    
    return prob_b

## Compute log-likelihood "THE STOCK MARKET FELL BY ONE HUNDRED POINTS LAST WEEK" 
def log_likelihood_u1(prob_u):
    sentence1="THE STOCK MARKET FELL BY ONE HUNDRED POINTS LAST WEEK"
    sentence1=sentence1.split(" ")
    L1=0
    for s in sentence1:
        if s in prob_u:
            L1 += np.log(prob_u[s])
    print("Log Likelihood of the unigram model of sentence1",L1)
    return L1
def log_likelihood_b1(prob_b):
    sentence1="<s> THE STOCK MARKET FELL BY ONE HUNDRED POINTS LAST WEEK"
    sentence1=sentence1.split(" ")     
    L2=0
    for i in range(len(sentence1) - 1):
        bi = (sentence1[i], sentence1[i + 1]) 
        if bi in prob_b:
            L2+=np.log(prob_b[bi])
    print("Log Likelihood of the bigram model of sentence1",L2)       
    
## Compute log-likelihood "THE SIXTEEN OFFICIALS SOLD FIRE INSURANCE"
def log_likelihood_u2(prob_u): 
    sentence2="THE SIXTEEN OFFICIALS SOLD FIRE INSURANCE"
    sentence2=sentence2.split(" ")
    L3=0
    for s in sentence2:
        if s in prob_u:
            L3 += np.log(prob_u[s])
    print("Log Likelihood of the unigram model of sentence2",L3)
def log_likelihood_b2(prob_b): 
    sentence2="<s> THE SIXTEEN OFFICIALS SOLD FIRE INSURANCE"
    sentence2=sentence2.split(" ")
    L4=0
    for i in range(len(sentence2) - 1):
        bi_2 = (sentence2[i], sentence2[i + 1]) 
        try:
            L4+=np.log(prob_b[bi_2])
        except:
            print("Log Likelihood of the bigram model of sentence2 = -inf")
def mixture_model(prob_u,prob_b,lam):
    L5= 0
    sentence3= "<s> THE SIXTEEN OFFICIALS SOLD FIRE INSURANCE"
    sentence3=sentence3.split(" ")
    for i in range(len(sentence3) - 1):
        word=sentence3[i + 1]
        pre_word=sentence3[i]
        m1= (pre_word, word) 
        if m1 in prob_b:
            prob_m = lam * prob_u[word] + (1 - lam) * prob_b[m1]
        else:
            prob_m = lam * prob_u[word]
        
        L5 += np.log(prob_m) 
              
    return L5
                   
vocb, unigram, bigram = load_file('hw4_vocab.txt', 'hw4_unigram.txt', 'hw4_bigram.txt')
prob_u = unigram_M(vocb, unigram)
prob_b=bigram_The(bigram)
u1=log_likelihood_u1(prob_u)
b1=log_likelihood_b1(prob_b)
u2=log_likelihood_u2(prob_u)
b2=log_likelihood_b2(prob_b)


lambda_values = np.linspace(0, 1, 100)
log_likelihoods = [mixture_model(prob_u, prob_b, lam) for lam in lambda_values]
optimal_lam=lambda_values[np.argmax(log_likelihoods)]
print(np.max(log_likelihoods))
print(optimal_lam)
plt.plot(lambda_values, log_likelihoods, label="Log-Likelihood ")
plt.xlabel("λ")
plt.ylabel("Log-Likelihood")
plt.title("Log-Likelihood Lm as a function of λ")
plt.legend()
plt.grid()
plt.show()
