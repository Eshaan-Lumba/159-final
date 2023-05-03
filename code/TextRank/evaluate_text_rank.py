
from rouge import Rouge
import nltk
from nltk.translate import bleu, meteor_score
from nltk.tokenize import word_tokenize
import sys
nltk.download('wordnet')
nltk.download('wordnet_ic')
nltk.download('punkt')


"""
Input: two sens of summarries to compare similarity
Returns: nothing, prints the bleu score for every summary pair
"""
def rougeValue(sen1, sen2):

    rouge = Rouge()
    return (rouge.get_scores(sen1, sen2))

"""
Method to run the analysis on a bunch of sentences. 
Input: lists of sens from prediction and reference, also the method
Return: the cumulative returnVal
"""
def getTotalVal(predList, refList, method):

    returnnValue = 0

    recall_u = 0
    precision_u = 0
    recall_b = 0
    precision_b = 0
    recall_f = 0
    precision_f = 0
    f1_u = 0
    f1_b = 0
    f1_f = 0
    metScore = 0


    for i in range(len(predList)):

        currentPred = predList[i]
        currentRef = refList[i]

        rouge_dict = rougeValue(currentPred, currentRef)
        recall_u += rouge_dict[0]['rouge-1']['r']
        precision_u += rouge_dict[0]['rouge-1']['p']
        f1_u += rouge_dict[0]['rouge-1']['f']
        
        recall_b += rouge_dict[0]['rouge-2']['r']
        precision_b += rouge_dict[0]['rouge-2']['p']
        f1_b += rouge_dict[0]['rouge-2']['f']
        
        recall_f += rouge_dict[0]['rouge-l']['r']
        precision_f += rouge_dict[0]['rouge-l']['p']
        f1_f += rouge_dict[0]['rouge-l']['f']

        ref = nltk.tokenize.word_tokenize(currentRef)
        hypo = nltk.tokenize.word_tokenize(currentPred)


        score = meteor_score.meteor_score([ref], hypo)
        metScore += score
        
    print("unigram f1: " + str(f1_u) + ", recall: " + str(recall_u) + ", precision: " + str(precision_u))
    print("bigram f1: " + str(f1_b) + ", recall: " + str(recall_b) + ", precision: " + str(precision_b))
    print("l f1: " + str(f1_f) + ", recall: " + str(recall_f) + ", precision: " + str(precision_f))

    print("Meteor: " + str(metScore))

    return returnnValue


def compare_summaries(file1, file2):
    pred_file = open(file1)
    ref_file = open(file2)

    preds = pred_file.readlines()
    refs = ref_file.readlines()

    if len(preds) != len(refs):
        print("Both files must be the same length")
        return

    getTotalVal(preds, refs, "U")




def main():
    #sumfile, stopfile
    compare_summaries(sys.argv[1],sys.argv[2])

if __name__ == '__main__':
    main()


# good study discussing evaluation metrics
# https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00373/100686/SummEval-Re-evaluating-Summarization-Evaluation
