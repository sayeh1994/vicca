import spacy
import numpy as np
nlp = spacy.load("en_core_web_md")

def sim_score(ref,gen):
    a = nlp(ref).similarity(nlp(gen))
    return a

def calculate_ratio(table):
    ratios = []
    maximum = []
    for column in table.T:  # Iterate over the columns of the table
        max_value = np.max(column)
        average_value = np.mean(column)
        if max_value + average_value != 0:
            ratio = max_value / (max_value + average_value)
        else:
            ratio=0
        ratios.append(ratio)
        
    return ratios
 
def similarity_score(ref_term, gen_term):
    # Calculate the similarity scores between each pair of terms
    similarity_scores = [[sim_score(ref, gen) for gen in gen_term] for ref in ref_term]
    ratio = calculate_ratio(np.array(similarity_scores))

    # Determine the length of the longest term in each list
    # avg_score = sum(sum(similarity_scores, [])) / (len(ref_term) * len(gen_term))
    return ratio