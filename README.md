# hydra-chess
Time for chess with transformers


## Pretraining 

### Datasets
 - Chesscom: 4.9 million games
 - Millionsbase: 3.4 million games
 - Combined: 8.3 million games

### Runs

#### One Objective
1. `2023-06-27-132331`: two epochs on Millionsbase, 49% validation
2. `2023-06-27-235552`: one epoch on Combined, 71% validation
3. `2023-06-28-091417`: two epochs on Combined, 80% validation


#### Dual Objective
1. `2023-07-09-053425-pt`: eight epochs on 1mil positions from Millionsbase


#### Megaset
Validation move loss after batch 120000: 33.6147
Validation move accuracy after batch 120000: 0.7541
Validation board loss after batch 120000: 0.6988
Validation board accuracy after batch 120000: 0.967




### Saving Commands


#### Encoder

cp -r /home/ubuntu/hydra-chess/models/encoder/hydra-pt /home/ubuntu/hydra-chess/models/encoder/hydra-pt-backup-75k-steps
cp -r /home/ubuntu/hydra-chess/models/encoder/hydra-pt /home/ubuntu/hydra-chess/models/encoder/hydra-pt-backup-125k-steps

cp -r /home/ubuntu/hydra-chess/models/encoder/hydra-ft-classify /home/ubuntu/hydra-chess/models/encoder/hydra-ft-classify-backup


cp -r /home/ubuntu/hydra-chess/models/encoder/hydra-ft-classify /home/ubuntu/hydra-chess/models/encoder/hydra-ft-classify-backup-9k
cp -r /home/ubuntu/hydra-chess/models/encoder/hydra-ft-classify /home/ubuntu/hydra-chess/models/encoder/hydra-ft-classify-backup-18k

cp -r /home/ubuntu/hydra-chess/models/encoder/hydra-ft-classify /home/ubuntu/hydra-chess/models/encoder/hydra-ft-classify-backup-9k-2

cp -r /home/ubuntu/hydra-chess/models/encoder/hydra-ft-ndcg /home/ubuntu/hydra-chess/models/encoder/hydra-ft-ndcg-backup



#### Decoder

cp -r /home/ubuntu/hydra-chess/models/decoder/hydra-pt /home/ubuntu/hydra-chess/models/decoder/hydra-pt-backup-120k-steps

cp -r /home/ubuntu/hydra-chess/models/decoder/hydra-ft-classify /home/ubuntu/hydra-chess/models/decoder/hydra-ft-classify-backup

