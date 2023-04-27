from rouge import Rouge
from nltk.translate import bleu

sent_a = 'Today is such a nice day'.split()
sent_b = 'Today is such a good day'.split()

"""
Input: two sens of summarries to compare similarity
Returns: the bleu score
"""
def bleuValue(sen1, sen2):

    currentPred = sen1.split()
    currentRef = sen2.split()

    return (bleu([currentPred], currentRef))

"""
Input: two sens of summarries to compare similarity
Returns: nothing, prints the bleu score for every summary pair
"""
def rougeValue(sen1, sen2):

    rouge = Rouge()
    return (rouge.get_scores(sen1, sen2))

"""
Input: two sentences
Returns: F1 score of the sentences. 
"""

def F1(sen1, sen2):

    rouge = rougeValue(sen1, sen2)
    bleu = bleuValue(sen1, sen2)

    return 2 * (rouge * bleu) / (rouge + bleu)

"""
Method to run the analysis on a bunch of sentences. 
Input: lists of sens from prediction and reference, also the method
Return: the cumulative returnVal
"""
def getTotalVal(predList, refList, method):

    returnnValue = 0

    for i in range(len(predList)):

        currentPred = predList[i]
        currentRef = refList[i]

        if (method == "bleu"): returnnValue += bleuValue(currentPred, currentRef)
        if (method == "rouge"): returnnValue += rougeValue(currentPred, currentRef)
        if (method == "f1"): returnnValue += F1(currentPred, currentRef)

    return returnnValue

# testing
sent_a = ['Today is such a nice day']
sent_b = ['Today is such a good day']

# List of predictions and references
predictions = ["hello there", "general kenobi"]
references = ["hello there", "general kenobi"]

# good study discussing evaluation metrics
# https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00373/100686/SummEval-Re-evaluating-Summarization-Evaluation

