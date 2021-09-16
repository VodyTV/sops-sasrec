# Sequence or Pseudo-Sequence? #
### An Analysis of Sequential Recommendation Datasets ###
![header_sequence_image](https://user-images.githubusercontent.com/24967046/133691042-17c1b127-1b64-4363-8919-1c0fa39e4543.png)

This repo contains our re-implementation and extension of SASRec [CITE] as well as code to download, process and format the datasets we use in our paper.

In order to get the data, run the notebook [link] and processed datasets will be saved into a `data/` directory. 

This code a rebuild of this repo https://github.com/kang205/SASRec

## Paper
TODO: Link to paper
TODO: Link to video

## How to Run

### Params

| Param      | Description | Example |
| --------------------- | ----------- | ------------------- | 
| --f-name              | File name of interaction dataset | `ml-1m` |
| --shuffle-sequence    | Whether to shuffle the input sequences | `True` or `False` | 
| --batch-size          | Batch size | 1024 | 
| --lr                  | Learning Rate | 0.001 | 
| --maxlen              | Maximum length of sequences | 200 | 
| --hidden-dim          | Hidden Dimension | 100 | 
| --num-blocks          | Number of attention blocks | 2 | 
| --num-epochs          | Number of epochs | 200 |
| --num-heads           | Number of heads | 2 |
| --dropout-rate        | Dropout Rate | 0.2 |  
| --seed                | Random Seed | 123 |
| --output-metrics-path | Path to save results | 'results/experiment_results.txt' |


### Shuffled / Unshuffled Experiments
```
python -m sas_rec.experiment --f-name 'ml_1m' --shuffle-sequence False --batch-size 128 --seed 101 --num-epochs 1 --output-metrics-path 'results/experiment_results.txt'
```

### Rating Experiments
Modified Experiments
```
python -m sas_rec.modified_experiment --f-name 'ml_1m_rating' --shuffle-sequence False --batch-size 128 --seed 101 --num-epochs 1 --output-metrics-path 'results/experiment_results.txt'
```

Joint loss experiment on movielens

```
python -m sas_rec.joint_loss_experiment --f-name 'ml-1m-rating' --shuffle-sequence False --batch-size 128 --seed 101 --num-epochs 20 --output-metrics-path 'results/experiment_results.txt'
```

## Citations

SASRec https://arxiv.org/abs/1808.09781
