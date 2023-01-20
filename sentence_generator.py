# Generating Sentences

import pandas as pd
import random
random.seed(2000)

matrix_NPs = ['John','Mary']

embedded_nouns = []
with open("textfiles/professions_filtered.txt",encoding="utf-8") as NPs:
    for line in NPs:
        embedded_nouns.append(line.casefold().strip())

embedded_nouns_filtered = random.sample(embedded_nouns, 60)

intensional_verbs = []
with open("textfiles/intensional_verbs.txt",encoding="utf-8") as int_verbs:
    for line in int_verbs:
        intensional_verbs.append(line.casefold().strip())

intensional_verbs_nf = []
with open("textfiles/intensional_verbs_nf.txt",encoding="utf-8") as int_verbs:
    for line in int_verbs:
        intensional_verbs_nf.append(line.casefold().strip())

perceptual_verbs = []
with open("textfiles/perceptual_verbs.txt",encoding="utf-8") as perc_verbs:
    for line in perc_verbs:
        perceptual_verbs.append(line.casefold().strip())

embedded_verbs = []
with open("textfiles/intransitive_verbs.txt",encoding="utf-8") as emb_verbs:
    for line in emb_verbs:
        embedded_verbs.append(line.casefold().strip())

embedded_verbs_filtered = random.sample(embedded_verbs, 30)

followup_verbs = ["met", "greeted", "liked"]

def word_to_sentence(words: list):
    string = ""
    for word in words:
        if word != None:
            string = string+word+" "
    string=string[:-1]+"."
    return string

def mark_query(phrase: str):
    return "_"+phrase+"_"


vowels = ['a','e','i','o','u']
deictic = "that"
pronoun = "[pronoun]"

dict_for_pd = {'sentence':[], 'query_type':[], 'matrix_NP': [], 'matrix_verb':[], 'matrix_verb_type':[],
                'followup_verb':[] ,'embedded_noun':[], 'embedded_verb':[], 'determiner_type':[]}


for subject in matrix_NPs:
    for noun in embedded_nouns_filtered:
        if noun[0] in vowels:
            indefinite = "an"
        else:
            indefinite = "a"
        for emb_verb in embedded_verbs_filtered:
            for matrix_verb_type in ["intensional", "intensional_nf", "perceptual"]:
                matrix_verb_list = {"intensional": intensional_verbs, "intensional_nf":intensional_verbs_nf, "perceptual":perceptual_verbs}[matrix_verb_type]
                if matrix_verb_type=="intensional_nf":
                    copula = "to be"
                elif matrix_verb_type=="perceptual":
                    copula = None
                else:
                    copula = "is"
                for matrix_verb in matrix_verb_list:
                    for determiner in [indefinite, deictic]:
                        det_type = {indefinite: 'indefinite', deictic: 'deictic'}[determiner]
                        for query in ["embedded_NP","matrix_NP"]:
                            if query == "embedded_NP":
                                s1 = word_to_sentence([subject, matrix_verb, mark_query(determiner+" "+noun), copula, emb_verb])
                            else:
                                s1 = word_to_sentence([mark_query(subject), matrix_verb, determiner+" "+noun, copula, emb_verb])
                            for followup_verb in followup_verbs:
                                s2 = word_to_sentence(["I",followup_verb,pronoun])
                                sentence = s1+" "+s2
                                dict_for_pd['sentence'].append(sentence)
                                dict_for_pd['query_type'].append(query)
                                dict_for_pd['matrix_NP'].append(subject)
                                dict_for_pd['matrix_verb'].append(matrix_verb)
                                dict_for_pd['matrix_verb_type'].append(matrix_verb_type)
                                dict_for_pd['embedded_noun'].append(noun)
                                dict_for_pd['embedded_verb'].append(emb_verb)
                                dict_for_pd['determiner_type'].append(det_type)
                                dict_for_pd['followup_verb'].append(followup_verb)


sentences_df = pd.DataFrame.from_dict(dict_for_pd)
q_size = sentences_df.shape[0]//4
q1 = sentences_df.iloc[:q_size,]
q2 = sentences_df.iloc[q_size:q_size*2,]
q3 = sentences_df.iloc[q_size*2:q_size*3,]
q4 = sentences_df.iloc[q_size*3:,]
for label in ["q1","q2","q3","q4"]:
    df = {"q1":q1,"q2":q2,"q3":q3,"q4":q4}[label]
    string = "csvfiles/"+label+"_preprocessed.csv"
    df.to_csv(string)



