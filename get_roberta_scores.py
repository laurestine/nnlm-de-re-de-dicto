from drddroberta.wsc import wsc_utils
from fairseq.models.roberta import RobertaModel
import pandas as pd
import time
from tqdm import tqdm
import os

roberta_wsc = RobertaModel.from_pretrained('roberta.large.wsc')

roberta_wsc.cuda()

q1_df = pd.read_csv("csvfiles/q1_preprocessed.csv")
q2_df = pd.read_csv("csvfiles/q2_preprocessed.csv")
q3_df = pd.read_csv("csvfiles/q3_preprocessed.csv")
q4_df = pd.read_csv("csvfiles/q4_preprocessed.csv")
qwise_dfs = {'q1':q1_df, 'q2':q2_df, 'q3':q3_df, 'q4':q4_df}

scores = []
q = os.getenv("quarter") # Set to 'q1', 'q2', 'q3' or 'q4' for corresponding chunks of data.

df = qwise_dfs[q]

for sentence in tqdm(df['sentence']):
    scores.append(roberta_wsc.disambiguate_pronoun(sentence)[1].item())

df['score'] = scores

filename = "csvfiles/"+q+"_processed.csv"
qwise_dfs[q].to_csv(filename)
