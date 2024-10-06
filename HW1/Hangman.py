import numpy as np

#open the hw1_word_counts_05-1.txt
with open("D:\Hangman\hw1_word_counts_05-1.txt",'r') as file:   
    content=file.readlines()
    
# Initialize empty lists
nums=[]
words=[]
prior_probability={}
# Extract words and numbers from content
for item in content:
    word,num=item.split()
    if len(word)==5:    
        words.append(word)
        nums.append(int(num))

#Convert lists to NumPy arrays
words=np.array(words)
nums=np.array(nums)

#Sanity check
top_fifteen=words[np.argsort(nums)[-15:][::-1]] # Sort and take the last 15, then reverse for descending order
print(top_fifteen)
lest_fourteen=words[np.argsort(nums)[:14]] # Sort and take the first 14
print(lest_fourteen)       
    
#prior probability
total=np.sum(nums) 
for i in range(len(nums)):
    prob=nums[i]/total
    prior_probability[words[i]]=prob  


def marginal(word,next_char,position):
    flag=False
    for i in position:      #Check if the character exist in the word in specific position 
        if word[i-1]==next_char: 
            flag=True
            return 1        # Return 1 indicating that the character was found.
    else:
        return 0            # Return 0 indicating that the character was not found.
#calculate the denominator of Bayes Rule   
def denominator(true_charac, true_positions, false_charac):
    false_positions=list(set([1,2,3,4,5])-set(true_positions)) # Calculate the positions that have not been guessed
    denominator=0
    for w in words:
        flag1=True                                  # Assume that all true characters are in the correct positions
        flag2=False                                 # Assume that false positions do not contain incorrect characters
        for i,charac in enumerate(true_charac):
            if w[true_positions[i]-1]!=charac:
                flag1=False                         # Set flag1 to False if a true character is in the wrong position
        # Check the false positions to ensure they do not contain guessed wrong or already guessed characters
        for i in false_positions:
            if (w[i - 1] in false_charac) or (w[i - 1] in true_charac): 
                 flag2 = True                      # Set flag2 to True if a false position has a wrong or repeated character
        if flag1 ==True and flag2 == False: 
            denominator += prior_probability[w]
    return denominator
#Bayes's Rule Application
def bayes(word,true_charac, true_positions, false_charac,denominator):
    false_positions=list(set([1,2,3,4,5])-set(true_positions))
    flag1=True                                  # Assume that all true characters are in the correct positions
    flag2=False                                 # Assume that false positions do not contain incorrect characters
    for i,charac in enumerate(true_charac):
        if word[true_positions[i]-1]!=charac:
            flag1=False                         # Set flag1 to False if a true character is in the wrong position
        # Check the false positions to ensure they do not contain guessed wrong or already guessed characters
    for i in false_positions:
        if (word[i - 1] in false_charac) or (word[i - 1] in true_charac): 
                flag2 = True                      # Set flag2 to True if a false position has a wrong or repeated character
    if flag1 ==True and flag2 == False:
        numerator=prior_probability[word]
    else:
        numerator=0
    return numerator / denominator 
#compute predicitive probability
def pred_prob(next_charac,true_charac, true_positions, false_charac):
    # Initialize the probability to zero
    prob=0
    # Calculate the denominator using the provided true characters and positions, and the false characters
    Denominator= denominator(true_charac, true_positions, false_charac) 
    for word in words:
        Marginal = marginal(word, next_charac,list(set([1,2,3,4,5])-set(true_positions)))
        # Check if the marginal probability is not zero
        if Marginal != 0:
            # Calculate the Bayes probability
            Bayes = bayes(word,true_charac, true_positions, false_charac, Denominator)
            # Update the total probability by adding the product of Marginal and Bayes probabilities
            prob += Marginal*Bayes
    return prob    
           
#TEST CASE:
correct_guess = [[], [], ["A", "S"], ["A", "S"], ["O"], [], ["D", "I"], ["D", "I"], ["U"]]
correct_pos = [[],[], [1, 5], [1, 5], [3], [], [1, 4], [1, 4], [2]]
incorrect_guess = [[], ["E", "A"], [], ["I"], ["A", "E", "M", "N", "T"], ["E", "O"], [], ["A"], ["A", "E", "I", "O", "S"]]   
alphabet=[]
# Loop through ASCII values for uppercase letters 'A' (65) to 'Z' (90)
for i in range(65,91):
    alphabet.append(chr(i))
#Iterate through each round of guesses
for i in range(len(correct_guess)):
    true_charac, true_positions, false_charac = correct_guess[i], correct_pos[i], incorrect_guess[i]
    max_prob = 0
    next_guess = ""

 # Iterate through alphabet, excluding already guessed characters
    for char in [item for item in alphabet if item not in true_charac and item not in false_charac]:
        prob = pred_prob(char, true_charac, true_positions, false_charac)
        if prob > max_prob:                      # If the current probability is greater than the maximum probability
            max_prob = prob                      # Update the maximum probability
            next_guess = char                    # Set the next guess to the current character
    print("The next best guess is", next_guess, "with probability", max_prob)
    