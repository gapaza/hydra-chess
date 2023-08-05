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

1. 
2. `2023-07-09-053425-pt`: eight epochs on 1mil positions from Millionsbase




#### Megaset
Validation move loss after batch 120000: 33.6147
Validation move accuracy after batch 120000: 0.7541
Validation board loss after batch 120000: 0.6988
Validation board accuracy after batch 120000: 0.967




### Saving Commands

cp -r /home/ubuntu/hydra-chess/models/hydra-family/hydra-base /home/ubuntu/hydra-chess/models/hydra-family/hydra-base-backup-2
cp -r /home/ubuntu/hydra-chess/models/hydra-family/hydra-full /home/ubuntu/hydra-chess/models/hydra-family/hydra-full-backup-2


cp -r /home/ubuntu/hydra-chess/models/hydra-family/hydra-full /home/ubuntu/hydra-chess/models/hydra-family/hydra-full-2

cp -r /home/ubuntu/hydra-chess/models/hydra-family/hydra-full-ft /home/ubuntu/hydra-chess/models/hydra-family/hydra-full-ft-backup


cp -r /home/ubuntu/hydra-chess/models/hydra-family/hydra-full-ft /home/ubuntu/hydra-chess/models/hydra-family/hydra-full-dn
cp -r /home/ubuntu/hydra-chess/models/hydra-family/hydra-full-ft /home/ubuntu/hydra-chess/models/hydra-family/hydra-full-dn2




cp -r /home/ubuntu/hydra-chess/models/hydra-family/hydra-full-enc-v3 /home/ubuntu/hydra-chess/models/hydra-family/hydra-full-enc-v3-backup



