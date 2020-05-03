from FeatureGenerator import *
import pandas as pd
import numpy as np
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

class SentimentFeatureGenerator(FeatureGenerator):


    def __init__(self, name='sentimentFeatureGenerator'):
        super(SentimentFeatureGenerator, self).__init__(name)


    def process(self, df, header="train"):

        print('generating sentiment features')
        print('for headline')
        
        # calculate the polarity score of each sentence then take the average
        sid = SentimentIntensityAnalyzer()
        def compute_sentiment(sentences):
            result = []
            for sentence in sentences:
                vs = sid.polarity_scores(sentence)
                result.append(vs)
            return pd.DataFrame(result).mean()
        
        df['headline_sents'] = df['Headline'].apply(lambda x: sent_tokenize(x))
        df = pd.concat([df, df['headline_sents'].apply(lambda x: compute_sentiment(x))], axis=1)
        df.rename(columns={'compound':'h_compound', 'neg':'h_neg', 'neu':'h_neu', 'pos':'h_pos'}, inplace=True)

        headlineSenti = df[['h_compound','h_neg','h_neu','h_pos']].values
        print('headlineSenti.shape:', headlineSenti.shape)
        
        outfilename_hsenti= "feature_pkl/"+header+".headline.senti.pkl"
        with open(outfilename_hsenti, "wb") as outfile:
            pickle.dump(headlineSenti, outfile, -1)
        print('headline sentiment features of dataset saved in %s' % outfilename_hsenti)
            
        print('headine senti done')
        
        print('for body')
        df['body_sents'] = df['articleBody'].map(lambda x: sent_tokenize(x))
        df = pd.concat([df, df['body_sents'].apply(lambda x: compute_sentiment(x))], axis=1)
        df.rename(columns={'compound':'b_compound', 'neg':'b_neg', 'neu':'b_neu', 'pos':'b_pos'}, inplace=True)
    
        bodySenti = df[['b_compound','b_neg','b_neu','b_pos']].values
        print('bodySenti.shape:', bodySenti.shape)
        
        outfilename_bsenti = "feature_pkl/"+header+".body.senti.pkl"
        with open(outfilename_bsenti, "wb") as outfile:
            pickle.dump(bodySenti, outfile, -1)
        print('body sentiment features of dataset saved in %s' % outfilename_bsenti)
        
        print('body senti done')

        return headlineSenti, bodySenti


    def read(self, header='train'):

        filename_hsenti = "feature_pkl/%s.headline.senti.pkl" % header
        with open(filename_hsenti, "rb") as infile:
            headlineSenti = pickle.load(infile)

        filename_bsenti = "feature_pkl/%s.body.senti.pkl" % header
        with open(filename_bsenti, "rb") as infile:
            bodySenti = pickle.load(infile)

        print('headlineSenti.shape:', headlineSenti.shape)
        print('bodySenti.shape:', bodySenti.shape)

        return [headlineSenti, bodySenti]

 #   Copyright 2017 Cisco Systems, Inc.
 #  
 #   Licensed under the Apache License, Version 2.0 (the "License");
 #   you may not use this file except in compliance with the License.
 #   You may obtain a copy of the License at
 #  
 #     http://www.apache.org/licenses/LICENSE-2.0
 #  
 #   Unless required by applicable law or agreed to in writing, software
 #   distributed under the License is distributed on an "AS IS" BASIS,
 #   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 #   See the License for the specific language governing permissions and
 #   limitations under the License.
