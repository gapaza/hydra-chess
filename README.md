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
cp -r /home/ubuntu/hydra-chess/models/encoder/hydra-pt /home/ubuntu/hydra-chess/models/encoder/hydra-pt-backup

cp -r /home/ubuntu/hydra-chess/models/encoder/hydra-pt /home/ubuntu/hydra-chess/models/encoder/hydra-pt-backup-75k-steps
cp -r /home/ubuntu/hydra-chess/models/encoder/hydra-pt /home/ubuntu/hydra-chess/models/encoder/hydra-pt-backup-118k-steps-large

cp -r /home/ubuntu/hydra-chess/models/encoder/hydra-ft-classify /home/ubuntu/hydra-chess/models/encoder/hydra-ft-classify-backup


cp -r /home/ubuntu/hydra-chess/models/encoder/hydra-ft-classify /home/ubuntu/hydra-chess/models/encoder/hydra-ft-classify-backup-9k
cp -r /home/ubuntu/hydra-chess/models/encoder/hydra-ft-classify /home/ubuntu/hydra-chess/models/encoder/hydra-ft-classify-backup-18k

cp -r /home/ubuntu/hydra-chess/models/encoder/hydra-ft-classify /home/ubuntu/hydra-chess/models/encoder/hydra-ft-classify-backup-9k-2



cp -r /home/ubuntu/hydra-chess/models/encoder/hydra-ft-classify /home/ubuntu/hydra-chess/models/encoder/hydra-ft-classify-large-1k

cp -r /home/ubuntu/hydra-chess/models/encoder/hydra-ft-classify /home/ubuntu/hydra-chess/models/encoder/hydra-ft-classify-large-2k

cp -r /home/ubuntu/hydra-chess/models/encoder/hydra-ft-classify /home/ubuntu/hydra-chess/models/encoder/hydra-ft-classify-large-9k


cp -r /home/ubuntu/hydra-chess/models/encoder/hydra-ft-ndcg /home/ubuntu/hydra-chess/models/encoder/hydra-ft-ndcg-backup



#### Decoder

cp -r /home/ubuntu/hydra-chess/models/decoder/hydra-pt /home/ubuntu/hydra-chess/models/decoder/hydra-pt-backup-120k-steps

cp -r /home/ubuntu/hydra-chess/models/decoder/hydra-ft-classify /home/ubuntu/hydra-chess/models/decoder/hydra-ft-classify-backup



#### Hybrid

cp -r /home/ubuntu/hydra-chess/models/hybrid/hydra-pt /home/ubuntu/hydra-chess/models/hybrid/hydra-pt-118k-steps




cp -r /home/ubuntu/hydra-chess/models/hybrid/hydra-ft-classify /home/ubuntu/hydra-chess/models/hybrid/hydra-ft-classify-18k-steps
cp -r /home/ubuntu/hydra-chess/models/hybrid/hydra-ft-classify /home/ubuntu/hydra-chess/models/hybrid/hydra-ft-classify-1k-steps

cp -r /home/ubuntu/hydra-chess/models/hybrid/hydra-ft-classify /home/ubuntu/hydra-chess/models/hybrid/hydra-ft-classify-2k-steps

cp -r /home/ubuntu/hydra-chess/models/hybrid/hydra-ft-ndcg /home/ubuntu/hydra-chess/models/hybrid/hydra-ft-ndcg-3k

cp -r /home/ubuntu/hydra-chess/models/hybrid/hydra-ft-ndcg /home/ubuntu/hydra-chess/models/hybrid/hydra-ft-ndcg-7k




### Distributed


cp -r /home/ubuntu/hydra-chess/models/hybrid/hydra-pt /home/ubuntu/hydra-chess/models/hybrid/hydra-pt-30k-steps
cp -r /home/ubuntu/hydra-chess/models/hybrid/hydra-pt /home/ubuntu/hydra-chess/models/hybrid/hydra-pt-60k-steps




### Distributed Window Testing

cp -r /home/ubuntu/hydra-chess/models/hybrid/hydra-pt /home/ubuntu/hydra-chess/models/hybrid/hydra-pt-5w-12k
cp -r /home/ubuntu/hydra-chess/models/hybrid/hydra-pt /home/ubuntu/hydra-chess/models/hybrid/hydra-pt-7w-12k
cp -r /home/ubuntu/hydra-chess/models/hybrid/hydra-pt /home/ubuntu/hydra-chess/models/hybrid/hydra-pt-9w-12k
cp -r /home/ubuntu/hydra-chess/models/hybrid/hydra-pt /home/ubuntu/hydra-chess/models/hybrid/hydra-pt-11w-12k


### True Unsupervised 

cp -r /home/ubuntu/hydra-chess/models/hybrid/hydra-pt /home/ubuntu/hydra-chess/models/hybrid/hydra-pt-9w-25k-uns
cp -r /home/ubuntu/hydra-chess/models/hybrid/hydra-pt /home/ubuntu/hydra-chess/models/hybrid/hydra-pt-9w-50k-uns
cp -r /home/ubuntu/hydra-chess/models/hybrid/hydra-pt /home/ubuntu/hydra-chess/models/hybrid/hydra-pt-9w-75k-uns
cp -r /home/ubuntu/hydra-chess/models/hybrid/hydra-pt /home/ubuntu/hydra-chess/models/hybrid/hydra-pt-9w-100k-uns


### True Unsupervised 12

cp -r /home/ubuntu/hydra-chess/models/hybrid/hydra-pt /home/ubuntu/hydra-chess/models/hybrid/hydra-pt-mw-25k-12l
cp -r /home/ubuntu/hydra-chess/models/hybrid/hydra-pt /home/ubuntu/hydra-chess/models/hybrid/hydra-pt-mw-50k-12l



### True Unsupervised 16

cp -r /home/ubuntu/hydra-chess/models/hybrid/hydra-pt /home/ubuntu/hydra-chess/models/hybrid/hydra-pt-mw-25k-16l


cp -r /home/ubuntu/hydra-chess/models/hybrid/hydra-ft-classify /home/ubuntu/hydra-chess/models/hybrid/hydra-ft-classify-16l-1k



