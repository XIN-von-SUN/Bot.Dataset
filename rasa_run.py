import rasa_data_process 
import rasa_NLU 
import pandas as pd


def rasa_nlu_loop(data_file, write_file, dimensions, range_dim):
    '''
    This function will generate the sub-dataset as training set based on the chosen dimensions and properties, 
    and then train the rasa NLU models based on these sub-datasets,
    Finally evaluate the models getting the corresponding evaluation metric values.
    ''' 
    eval_metrics = []
    model_set = []
    # sub-dataset part.
    for i in range(range_dim):  # range(len(dimensions))

        dim1 = dimensions[i][0]
        dim2 = dimensions[i][1] #if dimensions[i][1] else bool(dimensions[i][1])
        dim3 = dimensions[i][2] #if dimensions[i][2] else bool(dimensions[i][2])
        dim4 = dimensions[i][3] #if dimensions[i][3] else bool(dimensions[i][3])
        dim5 = dimensions[i][4]
        dim6 = dimensions[i][5]

        # In order to avoiding the same training model config.
        if [dim1,dim2,dim3,dim4,dim5,dim6] not in model_set:
            model_set.append([dim1,dim2,dim3,dim4,dim5,dim6])

            file_name = str(dim1)+'-'+str(dim2)+'-'+str(dim3)+'-'+str(dim4)+'-'+str(dim5)+'-'+str(dim6) 
            subset_file = write_file + file_name + '.md'
            rasa_data_process.rasa_data_loop(data_file, subset_file, dimensions[i])

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
            
            eval_metrics.append([file_name, dim1,dim2,dim3,dim4,dim5,dim6, eval_result, overall_accuracy, overall_f1_score, overall_precision])
    
            result = pd.DataFrame(eval_metrics, columns=['model_name', 'dim1_rm', 'dim2_sent_len', 'dim3_sent_num', 'dim4_pattern', 'dim5_SDPs', 'dim6_keywords', 'eval_result', 'overall_accuracy', 'overall_f1_score', 'overall_precision'])
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
    
    rasa_nlu_loop(data_file, write_file, dimensions, range_dim)