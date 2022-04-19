import torch
import json
import string
import csv
from transformers import AutoTokenizer, AutoModelForMaskedLM

print("Loading model and tokenizer.")

tokenizer = AutoTokenizer.from_pretrained("../input/roberta-finetuned-for-wsc/")
model = AutoModelForMaskedLM.from_pretrained("../input/roberta-finetuned-for-wsc/")

model.eval()
torch.no_grad()

print("Model and Tokenizer ready!")

#######################################################################
######## Validating that the imported model does well on WSC ##########
#######################################################################

# Get Winograd validation set
val_raw = open("../input/winograd/val.jsonl",encoding='utf-8')
masktoken = "<mask>"
(masktoken_id,) = set(tokenizer.encode(masktoken))-set([0,2])
pctn = string.punctuation

# Combines words into a sentence.
def make_sentence(words: list):
    running_string = ""
    for i in range(len(words)):
        if i < len(words)-1:
            if words[i+1] == ".":
                running_string = running_string+words[i]
            else:
                running_string = running_string+words[i]+" "
        else:
            running_string = running_string+words[i]
    return running_string

# Replaces pronoun with mask token in a test sentence.
def mask_given_dict(sentence, dictionary):
    pronoun_index = dictionary['target']['span2_index']
    s_as_list = sentence.split()
    s_as_list[pronoun_index] = masktoken
    return make_sentence(s_as_list)

print("Evaluating RoBERTa on WSC.")

# Loop through the validation set,
# feeding each one into the model.
list_for_checking = []
count = 0
num_correct = 0
for i in val_raw:
    jsonned = json.loads(i)
    text = jsonned['text']
    # We replace the pronoun to resolve with a mask token.
    masked_text = mask_given_dict(text,jsonned)
    pronoun = jsonned['target']['span2_text']
    tokenized = tokenizer(masked_text, return_tensors="pt")
    logits = model(**tokenized).logits
    mask_index = (tokenized['input_ids'][0]==masktoken_id).nonzero().item()
    # We check, among all of the other words in the sentence,
    # which one has the highest score to replace the mask token.
    candidates = [tokenizer.encode(x) for x in masked_text.split()]
    candidates = candidates + [tokenizer.encode(x) for x in [" "+y for y in masked_text.split()]]
    cset = set()
    for i in candidates:
        for j in i:
            cset.add(j)
    cset = cset - set([0,2,masktoken_id])
    notsoemptydict = dict()      
    for i in cset:
        notsoemptydict[i] = logits[:,mask_index,:][0][i]
    pred_word = tokenizer.decode(max(notsoemptydict, key=notsoemptydict.get))
    # We predict True if the selected word appears in the target span,
    # and False otherwise.
    if pred_word.strip() in jsonned['target']['span1_text']:
        pred_label = True
    else:
        pred_label = False
    if pred_label == jsonned['label']:
        num_correct = num_correct+1
    # print("Round ",'###',count,"###","over.")
    count = count+1
    list_for_checking.append([jsonned['target']['span1_text'],pred_word,jsonned['label'],pred_label])

print("Accuracy: ",list_for_checking/count)

#######################################################################
# Ensuring that we only select professions that tokenize as one word. #
#######################################################################

print("Processing professions.")

professions_raw = []
with open("../input/professions/professions.txt",encoding='utf-8') as f:
    for line in f:
        professions_raw.append(line)
for i in range(len(professions_raw)-1):
    professions_raw[i] = professions_raw[i][:-1]

professions_tokenized = {}
for i in professions_raw:
    professions_tokenized[i] = (tokenizer(i, return_tensors="pt"),tokenizer(' '+i, return_tensors="pt"))

professions_available_with_space = []
professions_available_no_space = []
professions_available_both = []
for i in professions_tokenized.keys():
    if professions_tokenized[i][0]['input_ids'].shape[1] == 3:
        professions_available_no_space.append(i)
        if professions_tokenized[i][1]['input_ids'].shape[1] == 3:
            professions_available_both.append(i)
    if professions_tokenized[i][1]['input_ids'].shape[1] == 3:
        professions_available_with_space.append(i)

with open("professions-filtered.txt",mode='w',encoding='utf-8') as f:
    for i in professions_available_with_space:
        f.write(i+'\n')

#######################################################################
##################### Generating test sentences #######################
#######################################################################

print("Generating test sentences.")

# Vowels used to determine if indefinite is "a" or "an".
vowels = ['a','e','i','o','u']

NPs_list = []
with open("../input/professions/professions-filtered.txt",encoding="utf-8") as NPs:
    for line in NPs:
        NPs_list.append(line.casefold().strip())

intensional_verbs_list = [[],[]]
with open("../input/verbs-for-use/intensional_verbs.txt",encoding="utf-8") as intensional_verbs:            
    for line in intensional_verbs:
        temp = line.split('/')
        intensional_verbs_list[0].append(temp[0].strip().casefold())
        intensional_verbs_list[1].append(temp[1].strip().casefold())

intensional_verbs_nf_list = [[],[]]
with open("../input/verbs-for-use/intensional_verbs_nf.txt",encoding="utf-8") as intensional_verbs_nf: 
    for line in intensional_verbs_nf:
        temp = line.split('/')
        intensional_verbs_nf_list[0].append(temp[0].strip().casefold())
        intensional_verbs_nf_list[1].append(temp[1].strip().casefold())

intransitive_verbs_list = []
with open("../input/verbs-for-use/intransitive_verbs.txt",encoding="utf-8") as intransitive_verbs: 
    for line in intransitive_verbs:
        intransitive_verbs_list.append(line.casefold().strip())

perceptual_verbs_list = [[],[]]
with open("../input/verbs-for-use/perceptual_verbs.txt",encoding="utf-8") as perceptual_verbs: 
    for line in perceptual_verbs:
        temp = line.split('/')
        perceptual_verbs_list[0].append(temp[0].strip())
        perceptual_verbs_list[1].append(temp[1].strip())

matrix_subjects = ["John","Mary"]

def word2sentence(words: list):
    string = ""
    for word in words:
        string = string+word+" "
    string=string[:-1]+"."
    return string

def copula_generator(tense, tensedlist):
    if tense == tensedlist[0]:
        return "is"
    if tense == tensedlist[1]:
        return "was"

def pst_or_prs(copula):
    if copula == "is":
        return "prs"
    if copula == "was":
        return "pst"

deictic = "that"
npi = "any"
dictlist_test = []
dictlist_deictic = []
dictlist_npi = []
dictlist_perceptual = []
count=0

# Code used to randomly select embedded subjects and embedded verbs:
''' Here's how we got these indices:
import random
itv_numbers = random.sample(range(len(intransitive_verbs_list)-1),18)
np_numbers = random.sample(range(len(NPs_list)-1),18)
> itv_numbers
[35, 18, 39, 23, 16, 46, 26, 41, 34, 33, 27, 2, 14, 32, 31, 3, 6, 42]
> np_numbers
[188, 161, 160, 72, 61, 6, 165, 123, 53, 35, 5, 195, 87, 11, 48, 187, 49, 175]
'''

# The resulting embedded subjects and embedded verbs:
itv_numbers = [35, 18, 39, 23, 16, 46, 26, 41, 34, 33, 27, 2, 14, 32, 31, 3, 6, 42]
np_numbers = [188, 161, 160, 72, 61, 6, 165, 123, 53, 35, 5, 195, 87, 11, 48, 187, 49, 175]
intransitive_verbs_list = [intransitive_verbs_list[i] for i in itv_numbers]
NPs_list = [NPs_list[i] for i in np_numbers]

# Generating a dictionary of sentences and their data:
for subject in matrix_subjects:
    for tense in intensional_verbs_list:
        for intransitive_verb in intransitive_verbs_list:
            for np in NPs_list:
                if np[0] in vowels:
                    article = "an"
                if np[0] not in vowels:
                    article = "a"
                for verb in tense:
                    s1 = word2sentence([subject,verb,article,np,
                                        copula_generator(tense, intensional_verbs_list),
                                        intransitive_verb])
                    s1_deictic = word2sentence([subject,verb,deictic,np,
                                        copula_generator(tense, intensional_verbs_list),
                                        intransitive_verb])
                    s1_npi = word2sentence([subject,verb,npi,np,
                                        copula_generator(tense, intensional_verbs_list),
                                        intransitive_verb])
                    s2 = "I met <mask>."
                    text_test = s1+" "+s2
                    text_deictic = s1_deictic+" "+s2
                    text_npi = s1_npi+" "+s2
                    ltest = text_test.split()
                    ldeictic = text_deictic.split()
                    lnpi = text_npi.split()
                    dict_test = {"type": "test", "text":text_test,
                                 "options":[subject,np],"mask_index":len(ltest)-1,
                                 "idx": count,"tense": pst_or_prs(copula_generator(tense, intensional_verbs_list)), 
                               "main_verb": verb, "emb_verb": intransitive_verb, "NP": np}
                    dict_deictic = {"type": "deictic", "text":text_deictic,
                                 "options":[subject,np],"mask_index":len(ldeictic)-1,
                                 "idx": count,"tense": pst_or_prs(copula_generator(tense, intensional_verbs_list)), 
                               "main_verb": verb, "emb_verb": intransitive_verb, "NP": np}
                    dict_npi = {"type": "deictic", "text":text_npi,
                                 "options":[subject,np],"mask_index":len(lnpi)-1,
                                 "idx": count,"tense": pst_or_prs(copula_generator(tense, intensional_verbs_list)), 
                               "main_verb": verb, "emb_verb": intransitive_verb, "NP": np}
                    dictlist_test.append(dict_test)
                    dictlist_deictic.append(dict_deictic)
                    dictlist_npi.append(dict_npi) 
                    count = count+1
                    
    for tense in intensional_verbs_nf_list:
        for intransitive_verb in intransitive_verbs_list:
            for np in NPs_list:
                if np[0] in vowels:
                    article = "an"
                if np[0] not in vowels:
                    article = "a"
                for verb in tense:
                    s1 = word2sentence([subject,verb,article,np,
                                        "to","be",
                                        intransitive_verb])
                    s1_deictic = word2sentence([subject,verb,deictic,np,
                                        "to","be",
                                        intransitive_verb])
                    s1_npi = word2sentence([subject,verb,npi,np,
                                        "to","be",
                                        intransitive_verb])
                    s2 = "I met <mask>."
                    text_test = s1+" "+s2
                    text_deictic = s1_deictic+" "+s2
                    text_npi = s1_npi+" "+s2
                    ltest = text_test.split()
                    ldeictic = text_deictic.split()
                    lnpi = text_npi.split()
                    dict_test = {"type": "test", "text":text_test,
                                 "options":[subject,np],"mask_index":len(ltest)-1,
                                 "idx": count,"tense": pst_or_prs(copula_generator(tense, intensional_verbs_nf_list)), 
                               "main_verb": verb, "emb_verb": intransitive_verb, "NP": np}
                    dict_deictic = {"type": "deictic", "text":text_deictic,
                                 "options":[subject,np],"mask_index":len(ldeictic)-1,
                                 "idx": count,"tense": pst_or_prs(copula_generator(tense, intensional_verbs_nf_list)), 
                               "main_verb": verb, "emb_verb": intransitive_verb, "NP": np}
                    dict_npi = {"type": "npi", "text":text_npi,
                                 "options":[subject,np],"mask_index":len(lnpi)-1,
                                 "idx": count,"tense": pst_or_prs(copula_generator(tense, intensional_verbs_nf_list)), 
                               "main_verb": verb, "emb_verb": intransitive_verb, "NP": np}
                    dictlist_test.append(dict_test)
                    dictlist_deictic.append(dict_deictic)
                    dictlist_npi.append(dict_npi)
                    count = count+1
                    
    for tense in perceptual_verbs_list:
        for intransitive_verb in intransitive_verbs_list:
            for np in NPs_list:
                if np[0] in vowels:
                    article = "an"
                if np[0] not in vowels:
                    article = "a"
                for verb in tense:
                    s1 = word2sentence([subject,verb,article,np,intransitive_verb])
                    s1_deictic = word2sentence([subject,verb,deictic,np,intransitive_verb])
                    s2 = "I met <mask>."
                    text_test = s1+" "+s2
                    text_deictic = s1_deictic+" "+s2
                    ltest = text_test.split()
                    ldeictic = text_deictic.split()
                    dict_test = {"type": "perc_reg", "text":text_test,
                                 "options":[subject,np],"mask_index":len(ltest)-1,
                                 "idx": count,"tense": pst_or_prs(copula_generator(tense, perceptual_verbs_list)), 
                               "main_verb": verb, "emb_verb": intransitive_verb, "NP": np}
                    dict_deictic = {"type": "perc_deictic", "text":text_deictic,
                                 "options":[subject,np],"mask_index":len(ldeictic)-1,
                                 "idx": count,"tense": pst_or_prs(copula_generator(tense, perceptual_verbs_list)), 
                               "main_verb": verb, "emb_verb": intransitive_verb, "NP": np}
                    dictlist_perceptual.append(dict_test)
                    dictlist_perceptual.append(dict_deictic)
                    count = count+1

print("Lists ready!")

with open('test_sentences.txt', 'w') as file:
    for sentence in dictlist_test:
        file.write(str(sentence)+'\n')
    
with open('deictic_sentences.txt', 'w') as file:
    for sentence in dictlist_deictic:
        file.write(str(sentence)+'\n')

with open('npi_sentences.txt', 'w') as file:
    for sentence in dictlist_npi:
        file.write(str(sentence)+'\n')
    
with open('perceptual_sentences.txt', 'w') as file:
    for sentence in dictlist_perceptual:
        file.write(str(sentence)+'\n')


#######################################################################
#################### Computing Scores of Sentences ####################
#######################################################################

# Helper functions for computing scores for each sentence in the data.
def compute_scores(dictionary: dict):
    text = dictionary['text']
    text_tokenized = tokenizer(text,return_tensors="pt")
    logits_text = model(**text_tokenized).logits
    mask_index = (text_tokenized['input_ids'][0]==masktoken_id).nonzero().item()
    options = dictionary['options']
    options_tokens = [tokenizer(" "+option)['input_ids'][1] for option in options]
    scores = [logits_text[:,mask_index,:][0][token] for token in options_tokens]
    return scores

header = ["index","type","text","matrix_subject","emb_subject",
          "matrix_verb","emb_verb","tense","matrix_subj_score","emb_subj_score"]

n_indices_reg = len(dictlist_test)
n_indices_perc = len(dictlist_perceptual)
quarter_reg = int(n_indices_reg/4)
quarter_perc = int(n_indices_perc/4)

# Data is split into quarters to more easily run with limited GPU time.

print("Generating model scores.")

#################### QUARTER 1 ######################
print("Computing on the first quarter of data.")
with open('q1_of_data.csv', 'w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for i in range(quarter_reg):
        if i == 50:
            print("reached 50!")
        if i == 1000:
            print("reached 1k!")
        if i == 5000:
            print("reached 5k!")
        if i == 10000:
            print("reached 10k!")
        dict_test = dictlist_test[i]
        dict_deictic = dictlist_deictic[i]
        dict_npi = dictlist_npi[i]
        internal_dictlist = [dict_test,dict_deictic,dict_npi]
        scores = (compute_scores(dict_test),compute_scores(dict_deictic),compute_scores(dict_npi))
        rows = []
        for j in range(len(internal_dictlist)):
            rows.append([
                        internal_dictlist[j]['idx'], internal_dictlist[j]['type'],
                        internal_dictlist[j]['text'],internal_dictlist[j]['options'][0],
                        internal_dictlist[j]['options'][1],internal_dictlist[j]['main_verb'],
                        internal_dictlist[j]['emb_verb'],internal_dictlist[j]['tense'],
                        scores[j][0],scores[j][1]
                        ])
        writer.writerows(rows)
    for i in range(quarter_perc):
        if i == 50:
            print("reached 50!")
        if i == 1000:
            print("reached 1k!")
        if i == 5000:
            print("reached 5k!")
        dictionary = dictlist_perceptual[i]
        scores = compute_scores(dictionary)
        writer.writerow([
                        dictionary['idx'], dictionary['type'], 
                        dictionary['text'], dictionary['options'][0],
                        dictionary['options'][1], dictionary['main_verb'],
                        dictionary['emb_verb'], dictionary['tense'],
                        scores[0],scores[1]
                        ])

#################### QUARTER 2 ######################
print("Computing on the second quarter of data.")
with open('q2_of_data.csv', 'w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for i in range(quarter_reg,(quarter_reg*2)):
        if i == quarter_reg + 50:
            print("reached 50!")
        if i == quarter_reg + 1000:
            print("reached 1k!")
        if i == quarter_reg + 5000:
            print("reached 5k!")
        if i == quarter_reg + 10000:
            print("reached 10k!")
        dict_test = dictlist_test[i]
        dict_deictic = dictlist_deictic[i]
        dict_npi = dictlist_npi[i]
        internal_dictlist = [dict_test,dict_deictic,dict_npi]
        scores = (compute_scores(dict_test),compute_scores(dict_deictic),compute_scores(dict_npi))
        rows = []
        for j in range(len(internal_dictlist)):
            rows.append([
                        internal_dictlist[j]['idx'], internal_dictlist[j]['type'],
                        internal_dictlist[j]['text'],internal_dictlist[j]['options'][0],
                        internal_dictlist[j]['options'][1],internal_dictlist[j]['main_verb'],
                        internal_dictlist[j]['emb_verb'],internal_dictlist[j]['tense'],
                        scores[j][0],scores[j][1]
                        ])
        writer.writerows(rows)
    for i in range(quarter_perc,(quarter_perc*2)):
        if i == quarter_perc + 50:
            print("reached 50!")
        if i == quarter_perc + 1000:
            print("reached 1k!")
        if i == quarter_perc + 5000:
            print("reached 5k!")
        dictionary = dictlist_perceptual[i]
        scores = compute_scores(dictionary)
        writer.writerow([
                        dictionary['idx'], dictionary['type'], 
                        dictionary['text'], dictionary['options'][0],
                        dictionary['options'][1], dictionary['main_verb'],
                        dictionary['emb_verb'], dictionary['tense'],
                        scores[0],scores[1]
                        ])


#################### QUARTER 3 ######################
print("Computing on the third quarter of data.")
with open('q3_of_data.csv', 'w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for i in range((quarter_reg*2),(quarter_reg*3)):
        if i == quarter_reg*2 + 50:
            print("reached 50!")
        if i == quarter_reg*2 + 1000:
            print("reached 1k!")
        if i == quarter_reg*2 + 5000:
            print("reached 5k!")
        if i == quarter_reg*2 + 10000:
            print("reached 10k!")
        dict_test = dictlist_test[i]
        dict_deictic = dictlist_deictic[i]
        dict_npi = dictlist_npi[i]
        internal_dictlist = [dict_test,dict_deictic,dict_npi]
        scores = (compute_scores(dict_test),compute_scores(dict_deictic),compute_scores(dict_npi))
        rows = []
        for j in range(len(internal_dictlist)):
            rows.append([
                        internal_dictlist[j]['idx'], internal_dictlist[j]['type'],
                        internal_dictlist[j]['text'],internal_dictlist[j]['options'][0],
                        internal_dictlist[j]['options'][1],internal_dictlist[j]['main_verb'],
                        internal_dictlist[j]['emb_verb'],internal_dictlist[j]['tense'],
                        scores[j][0],scores[j][1]
                        ])
        writer.writerows(rows)
    for i in range((quarter_perc*2),(quarter_perc*3)):
        if i == quarter_perc*2 + 50:
            print("reached 50!")
        if i == quarter_perc*2 + 1000:
            print("reached 1k!")
        if i == quarter_perc*2 + 5000:
            print("reached 5k!")
        dictionary = dictlist_perceptual[i]
        scores = compute_scores(dictionary)
        writer.writerow([
                        dictionary['idx'], dictionary['type'], 
                        dictionary['text'], dictionary['options'][0],
                        dictionary['options'][1], dictionary['main_verb'],
                        dictionary['emb_verb'], dictionary['tense'],
                        scores[0],scores[1]
                        ])
    
     
#################### QUARTER 4 ######################
print("Computing on the fourth quarter of data.")
with open('q4_of_data.csv', 'w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for i in range((quarter_reg*3),(quarter_reg*4)):
        if i == quarter_reg*3 + 50:
            print("reached 50!")
        if i == quarter_reg*3 + 1000:
            print("reached 1k!")
        if i == quarter_reg*3 + 5000:
            print("reached 5k!")
        if i == quarter_reg*3 + 10000:
            print("reached 10k!")
        dict_test = dictlist_test[i]
        dict_deictic = dictlist_deictic[i]
        dict_npi = dictlist_npi[i]
        internal_dictlist = [dict_test,dict_deictic,dict_npi]
        scores = (compute_scores(dict_test),compute_scores(dict_deictic),compute_scores(dict_npi))
        rows = []
        for j in range(len(internal_dictlist)):
            rows.append([
                        internal_dictlist[j]['idx'], internal_dictlist[j]['type'],
                        internal_dictlist[j]['text'],internal_dictlist[j]['options'][0],
                        internal_dictlist[j]['options'][1],internal_dictlist[j]['main_verb'],
                        internal_dictlist[j]['emb_verb'],internal_dictlist[j]['tense'],
                        scores[j][0],scores[j][1]
                        ])
        writer.writerows(rows)
    for i in range((quarter_perc*3),(quarter_perc*4)):
        if i == quarter_perc*3 + 50:
            print("reached 50!")
        if i == quarter_perc*3 + 1000:
            print("reached 1k!")
        if i == quarter_perc*3 + 5000:
            print("reached 5k!")
        dictionary = dictlist_perceptual[i]
        scores = compute_scores(dictionary)
        writer.writerow([
                        dictionary['idx'], dictionary['type'], 
                        dictionary['text'], dictionary['options'][0],
                        dictionary['options'][1], dictionary['main_verb'],
                        dictionary['emb_verb'], dictionary['tense'],
                        scores[0],scores[1]
                        ])

print("Finished!")
