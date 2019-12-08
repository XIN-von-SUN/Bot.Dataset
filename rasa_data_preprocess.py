import pandas as pd
import numpy as np

def rasa_data_preprocess(read_file, write_file):
    # read_file = '/users/xinsun/Downloads/oos-eval-master/data/data_full.json'
    df = pd.read_json(read_file, typ='series')

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


if __name__=='__main__':

    read_file = 
    write_file =
    rasa_data_preprocess(read_file, write_file)


