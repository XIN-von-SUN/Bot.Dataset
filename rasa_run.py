import rasa_data_process 
import rasa_NLU 
import pandas as pd


def rasa_nlu_loop(data_file, write_file, dimensions, properties):
    '''
    This function will generate the sub-dataset as training set based on the chosen dimensions and properties, 
    and then train the rasa NLU models based on these sub-datasets,
    Finally evaluate the models getting the corresponding evaluation metric values.
    '''
    eval_metrics = []
    # sub-dataset part.
    for i in range(1):  # range(len(dimensions))
        for j in range(len(properties)):   
            file_name = str(int(dimensions[i][0])) + str(int(dimensions[i][1])) + str(int(dimensions[i][2])) + str(int(dimensions[i][3])) + '-' + str(properties[j][0]) + '-' + str(properties[j][1]) + '-' + str(properties[j][2])
            subset_file = write_file + file_name + '.md'
            rasa_data_process.rasa_data_loop(data_file, subset_file, dimensions[i], properties[j])

            # rasa NLU models training part.
            model_directory, model_name = "./models/nlu/", file_name
            train_data_path, test_data_path = subset_file, "./data/data_test.md"
            train_loop = True

            rasa_nlu = rasa_NLU.rasa_NLU(train_loop, model_directory, model_name, train_data_path, test_data_path)
            
            if rasa_nlu.train_loop:
                print("Start training......")
                rasa_nlu.NLU_train()
            
            # rasa NLU models evaluation part.
            eval_result, overall_accuracy, overall_f1_score, overall_precision = rasa_nlu.NLU_evaluation()

            eval_metrics.append([file_name, eval_result, overall_accuracy, overall_f1_score, overall_precision])
    
            result = pd.DataFrame(eval_metrics, columns=['model_name', 'eval_result', 'overall_accuracy', 'overall_f1_score', 'overall_precision'])
            result.to_csv('/users/xinsun/Desktop/rasa_data/rasa_eval_result.csv')
            


if __name__=='__main__':

    data_file = '/users/xinsun/Downloads/oos-eval-master/data/data_full.json'
    write_file = '/users/xinsun/Desktop/rasa_data/'    #Downloads/oos-eval-master/data/data_full.json'

    dimensions = [[True,True,True,True], [True,True,True,False], [True,True,False,True], [True,True,False,False], 
                    [True,False,True,True], [True,False,True,False], [True,False,False,True], [True,False,False,False],
                    [False,True,True,True], [False,True,True,False], [False,True,False,True], [False,True,False,False],
                    [False,False,True,True], [False,False,True,False], [False,False,False,True], [False,False,False,False]] 
                    # dim_1, dim_2, dim_3, dim_4

    properties = [[5, 50, 'question'], [5, 50, 'statement'], [5, 100, 'question'], [5, 100, 'statement'], 
                    [10, 50, 'question'], [10, 50, 'statement'],
                    [15, 20, 'question'], [15, 20, 'statement'], [15, 50, 'question'], [15, 50, 'statement']]
                    # dim2_lenth, dim3_reduce, dim4_grammar
    
    rasa_nlu_loop(data_file, write_file, dimensions, properties)