import pandas as pd
import numpy as np
import string, re
import spacy
import random


class rasa_data:

    def __init__(self, data_file, dim_1, dim_2, dim_3, dim_4):
        self.data_file = data_file
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.dim_3 = dim_3
        self.dim_4 = dim_4



    def rasa_data_oov(self):
        pass



    def rasa_data_process(self, df, write_file):
        '''
        # read_file = '/users/xinsun/Downloads/oos-eval-master/data/data_full.json'
        df = pd.read_json(self.data_file, typ='series')

        df_oos_train = pd.DataFrame(df['oos_train'], columns=['Phrase', 'Intent'])
        df_oos_test = pd.DataFrame(df['oos_test'], columns=['Phrase', 'Intent'])
        df_oos_val = pd.DataFrame(df['oos_val'], columns=['Phrase', 'Intent'])
        df_train = pd.DataFrame(df['train'], columns=['Phrase', 'Intent'])
        df_test = pd.DataFrame(df['test'], columns=['Phrase', 'Intent'])
        df_val = pd.DataFrame(df['val'], columns=['Phrase', 'Intent'])

        df_train['Phrase'] = '- ' + df_train.Phrase + '.\n'
        df_test['Phrase'] = '- ' + df_test.Phrase + '.\n'

        intents_train = list(df_train.Intent.unique())
        intents_test = list(df_test.Intent.unique())

        data_train = []
        for intent in intents_train:
            data_train.append('## intent:' + str(intent) + '\n')
            phrase = df_train[df_train.Intent==intent]['Phrase'].values
            phrase_txt = str()
            for i in phrase: phrase_txt += str(i)
            data_train.append(phrase_txt + '\n')
            
        data_test = []
        for intent in intents_test:
            data_test.append('## intent:' + str(intent) + '\n')
            phrase = df_test[df_test.Intent==intent]['Phrase'].values
            phrase_txt = str()
            for i in phrase: phrase_txt += str(i)
            data_test.append(phrase_txt + '\n')

        # with open('data_test.txt', 'w') as f:   
        with open(write_file, 'w') as f:   
            for i in data_test:
                f.write(str(i))  
        '''

        df['Phrase'] = '- ' + df.Phrase + '.\n'
        intents = list(df.Intent.unique())

        data = []
        for intent in intents:
            data.append('## intent:' + str(intent) + '\n')
            phrase = df[df.Intent==intent]['Phrase'].values
            phrase_txt = str()
            for i in phrase: phrase_txt += str(i)
            data.append(phrase_txt + '\n')
        
        with open(write_file, 'w') as f:   
            for i in data:
                f.write(str(i))  



    def rasa_sub_dataset(self, write_file, dim2_length, dim3_reduce, dim4_grammar):
        #data_file = '/users/xinsun/Downloads/oos-eval-master/data/data_full.json'
        df = pd.read_json(self.data_file, typ='series')
        df_train = pd.DataFrame(df['train'], columns=['Phrase', 'Intent'])

        if self.dim_1:
            text = df_train.Phrase.values
            regex = re.compile('[%s]' % re.escape(string.punctuation))
            token_no_punctuation = [regex.sub('', i) for i in text]
            df_train['Phrase'] = token_no_punctuation

        if self.dim_2:
            nlp = spacy.load('en')
            phrases = df_train['Phrase']
            tokens = nlp.pipe(phrases, batch_size=10000)

            phrase_tokens = []
            for phrase in tokens: phrase_tokens.append(phrase)

            df_train_length = []
            for i in phrase_tokens: df_train_length.append(str(i[:dim2_length])) 

            df_train['Phrase'] = df_train_length

        if self.dim_3:
            idx_range= [(i*100, i*100+99) for i in range(150)]
            idx = []
            for i in range(150):
                idx.append([random.randint(idx_range[i][0], idx_range[i][1]) for _ in range(dim3_reduce)])

            index = []
            for i in idx: 
                for j in range(len(i)): index.append(i[j])

            df_train['Phrase_copy'] = df_train['Phrase']
            df_train['Phrase'] = None
            df_train.loc[index, 'Phrase'] = df_train.loc[index, 'Phrase_copy']
            df_train = df_train.drop(columns=['Phrase_copy'])

        if self.dim_4:
            df_train = df_train[df_train.Phrase.isnull()==False].reset_index(drop=True)
            nlp = spacy.load('en')
            phrases = df_train['Phrase']
            tokens = nlp.pipe(phrases, batch_size=10000)

            phrase_tokens = []
            for phrase in tokens: phrase_tokens.append(phrase)

            idx_question = []
            for i in range(len(df_train)):
                if str(phrase_tokens[i][0]) in ['what', 'why', 'who', 'how', 'can']:
                    idx_question.append(i)
            idx_statement = set(range(len(df_train))) - set(idx_question)

            df_train['Phrase_question'] = df_train.loc[idx_question, 'Phrase']
            df_train['Phrase_statement'] = df_train.loc[idx_statement, 'Phrase']

            if dim4_grammar == 'question': df_train['Phrase'] = df_train['Phrase_question']
            else: df_train['Phrase'] = df_train['Phrase_statement']

        df_final = df_train[df_train.Phrase.isnull()==False].reset_index(drop=True)
        #print(df_final)
        
        self.rasa_data_process(df_final[['Phrase', 'Intent']], write_file) 



def rasa_data_loop(data_file, write_file, dimensions, properties):

    rasa_dataset = rasa_data(data_file, dimensions[0], dimensions[1], dimensions[2], dimensions[3])
    rasa_dataset.rasa_sub_dataset(write_file, properties[0], properties[1], properties[2])




if __name__=='__main__':

    data_file = '/users/xinsun/Downloads/oos-eval-master/data/data_full.json'
    write_file = '/users/xinsun/Desktop/'    #Downloads/oos-eval-master/data/data_full.json'
    
    dimensions = [[True,True,True,True], [True,True,True,False], [True,True,False,True], [True,True,False,False], 
                    [True,False,True,True], [True,False,True,False], [True,False,False,True], [True,False,False,False],
                    [False,True,True,True], [False,True,True,False], [False,True,False,True], [False,True,False,False],
                    [False,False,True,True], [False,False,True,False], [False,False,False,True], [False,False,False,False]] 
                    # dim_1, dim_2, dim_3, dim_4

    properties = [[5, 20, 'question'], [5, 20, 'statement'], [5, 50, 'question'], [5, 50, 'statement'], [5, 100, 'question'], [5, 100, 'statement'], 
                    [10, 20, 'question'], [10, 20, 'statement'], [10, 50, 'question'], [10, 50, 'statement'], [10, 100, 'question'], [10, 100, 'statement'],
                    [15, 20, 'question'], [15, 20, 'statement'], [15, 50, 'question'], [15, 50, 'statement'], [15, 100, 'question'], [15, 100, 'statement']]
    # dim2_length, dim3_reduce, dim4_grammar

    for i in range(2):
        for j in range(2):
            file_name = str(int(dimensions[i][0])) + str(int(dimensions[i][1])) + str(int(dimensions[i][2])) + str(int(dimensions[i][3])) + '-' + str(properties[j][0]) + '-' + str(properties[j][1]) + '-' + str(properties[j][2])
                    
            subset_file = write_file + file_name + '.txt'
            print(subset_file)
            rasa_data_loop(data_file, subset_file, dimensions[i], properties[j])