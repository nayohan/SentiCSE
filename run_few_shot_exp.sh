# one-shot settings
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m bert-base-uncased -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m bert-base-uncased -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m bert-base-uncased -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m bert-base-uncased -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m princeton-nlp/sup-simcse-bert-base-uncased -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m princeton-nlp/sup-simcse-bert-base-uncased -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m princeton-nlp/sup-simcse-bert-base-uncased -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m princeton-nlp/sup-simcse-bert-base-uncased -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiBERT_bert -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiBERT_bert -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiBERT_bert -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiBERT_bert -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiX-base_bert -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiX-base_bert -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiX-base_bert -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiX-base_bert -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m google/electra-base-discriminator -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m google/electra-base-discriminator -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m google/electra-base-discriminator -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m google/electra-base-discriminator -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiWSP-base_electra -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiWSP-base_electra -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiWSP-base_electra -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiWSP-base_electra -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m roberta-base -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m roberta-base -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m roberta-base -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m roberta-base -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m princeton-nlp/sup-simcse-roberta-large -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m princeton-nlp/sup-simcse-roberta-large -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m princeton-nlp/sup-simcse-roberta-large -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m princeton-nlp/sup-simcse-roberta-large -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiLARE_roberta -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiLARE_roberta -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiLARE_roberta -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiLARE_roberta -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/imdb-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e130-b16-lr0.00001 -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/imdb-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e130-b16-lr0.00001 -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/imdb-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e130-b16-lr0.00001 -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/imdb-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e130-b16-lr0.00001 -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/yelp-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e6-b16-lr0.00001 -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/yelp-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e6-b16-lr0.00001 -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/yelp-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e6-b16-lr0.00001 -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/yelp-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e6-b16-lr0.00001 -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/sst-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e35-b16-lr0.00001 -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/sst-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e35-b16-lr0.00001 -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/sst-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e35-b16-lr0.00001 -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/sst-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e35-b16-lr0.00001 -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/mr-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e300-b16-lr0.00001 -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/mr-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e300-b16-lr0.00001 -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/mr-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e300-b16-lr0.00001 -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 2 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/mr-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e300-b16-lr0.00001 -gpu 3 -d imdb &
wait


# few-shot settings
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m bert-base-uncased -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m bert-base-uncased -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m bert-base-uncased -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m bert-base-uncased -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m princeton-nlp/sup-simcse-bert-base-uncased -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m princeton-nlp/sup-simcse-bert-base-uncased -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m princeton-nlp/sup-simcse-bert-base-uncased -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m princeton-nlp/sup-simcse-bert-base-uncased -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiBERT_bert -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiBERT_bert -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiBERT_bert -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiBERT_bert -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiX-base_bert -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiX-base_bert -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiX-base_bert -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiX-base_bert -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m google/electra-base-discriminator -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m google/electra-base-discriminator -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m google/electra-base-discriminator -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m google/electra-base-discriminator -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiWSP-base_electra -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiWSP-base_electra -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiWSP-base_electra -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiWSP-base_electra -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m roberta-base -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m roberta-base -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m roberta-base -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m roberta-base -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m princeton-nlp/sup-simcse-roberta-large -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m princeton-nlp/sup-simcse-roberta-large -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m princeton-nlp/sup-simcse-roberta-large -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m princeton-nlp/sup-simcse-roberta-large -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiLARE_roberta -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiLARE_roberta -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiLARE_roberta -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m ../pretrained_models/SentiLARE_roberta -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/imdb-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e130-b16-lr0.00001 -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/imdb-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e130-b16-lr0.00001 -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/imdb-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e130-b16-lr0.00001 -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/imdb-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e130-b16-lr0.00001 -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/yelp-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e6-b16-lr0.00001 -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/yelp-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e6-b16-lr0.00001 -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/yelp-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e6-b16-lr0.00001 -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/yelp-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e6-b16-lr0.00001 -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/sst-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e35-b16-lr0.00001 -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/sst-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e35-b16-lr0.00001 -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/sst-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e35-b16-lr0.00001 -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/sst-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e35-b16-lr0.00001 -gpu 3 -d imdb &
wait

python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/mr-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e300-b16-lr0.00001 -gpu 0 -d rotten_tomatoes &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/mr-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e300-b16-lr0.00001 -gpu 1 -d yelp_polarity &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/mr-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e300-b16-lr0.00001 -gpu 2 -d sst2 &
python few_shot_experiment.py -run 5 -t_num 10 -v_num 500  -epoch 100 -bs 10 -e_step 20 -lr 0.00001 -m /home/uj-user/SenCSE/result_simcse_old/acl_exp17_dataset_ablation/mr-mlm0.15-w0-sentivocab0.1-sent-roberta-base-e300-b16-lr0.00001 -gpu 3 -d imdb &
wait