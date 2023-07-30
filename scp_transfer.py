import config
import os
import paramiko
from scp import SCPClient



# scp -i ~/keys/gabe-master.pem /Users/gapaza/repos/gabe/hydra_old-chess/datasets/pt/millionsbase/millionsbase.zip ubuntu@3.145.44.57:/home/ubuntu/hydra_old-chess/datasets/pt/millionsbase
# scp -i ~/keys/gabe-master.pem /Users/gapaza/Downloads/stockfish-ubuntu-x86-64-avx2.tar ec2-user@18.218.228.158:/home/ec2-user/stockfish-15-dataset/stockfish
# scp -i ~/gabe-master.pem /home/ec2-user/stockfish-15-dataset/datasets/dataset_prob.pkl ubuntu@3.131.38.221:/home/ubuntu/hydra_old-chess/datasets/pt/


key_location = '~/keys/gabe-master.pem'
remote_user = 'ubuntu'
remote_address = '18.191.150.190'
file_name = 'lichess_ft'
run_type = 'ft'

remote_location = '/home/ubuntu/hydra_old-chess/datasets/'+run_type+'/' + file_name

if run_type == 'pt':
    transfer_file = os.path.join(config.pt_datasets_dir, file_name, file_name + '.zip')
elif run_type == 'ft':
    transfer_file = os.path.join(config.ft_datasets_dir, file_name, file_name + '.zip')
elif run_type == 'dc':
    transfer_file = os.path.join(config.ft_datasets_dir, file_name, file_name + '.zip')



def main():
    command = 'scp ' + '-i ' + key_location + ' ' + transfer_file + ' ' + remote_user + '@' + remote_address + ':' + remote_location
    print('SCP Command:', command)



def to_client():
    transfer_file = '/Users/gapaza/repos/gabe/hydra_old-chess/models/hybrid'
    remote_location = '/home/ubuntu/hydra_old-chess/models/hybrid/hydra_old-ft-ndcg-3k'
    command = 'scp -i ' + key_location + ' -r ' + remote_user + '@' + remote_address + ':' + remote_location + ' ' + transfer_file
    print('SCP Command:', command)
 # scp -i ~/keys/gabe-master.pem /Users/gapaza/repos/gabe/hydra_old-chess/datasets/ft/lc0_standard/lc0_standard.zip ubuntu@18.189.193.147:/home/ubuntu/hydra_old-chess/datasets/ft/lc0_standard



if __name__ == '__main__':
    main()
    # to_client()