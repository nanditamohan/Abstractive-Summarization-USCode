import string
import json

with open('./billsum_v4_1/us_train_data_final_OFFICIAL.json') as json_file:
    data = []
    for line in json_file: 
        data.append(json.loads(line))

    texts = ""
    summaries = ""
    for law in data: 
        texts += '\n' + law['text'].replace('\n','')
        summaries += '\n' + law['summary'].replace('\n','')
        
    train_story = open('./train_story.txt', 'w')
    train_story.write(texts)
    train_story.close()

    train_sum = open('./train_summ.txt', 'w')
    train_sum.write(summaries)
    train_sum.close()

with open('./billsum_v4_1/us_test_data_final_OFFICIAL.json') as json_eval:
    data_eval = []
    for line_eval in json_eval: 
        data_eval.append(json.loads(line_eval))

    texts_eval = ""
    summaries_eval = ""
    for law_eval in data_eval: 
        texts_eval += '\n' + law_eval['text'].replace('\n','')
        summaries_eval += '\n' + law_eval['summary'].replace('\n','')
        
    eval_story = open('./eval_story.txt', 'w')
    eval_story.write(texts_eval)
    eval_story.close()

    eval_sum = open('./eval_summ.txt', 'w')
    eval_sum.write(summaries_eval)
    eval_sum.close()
