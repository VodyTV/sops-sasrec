# sas-rec
Our implementation of SASRec


## How to Run

Joint loss experiment on movielens

`python -m sas_rec.joint_loss_experiment --f-name 'ml-1m-rating' --shuffle-sequence False --batch-size 128 --seed 101 --num-epochs 20 --output-metrics-path 'results/sasrec_outs.txt'`
