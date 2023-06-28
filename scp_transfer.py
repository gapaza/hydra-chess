import config
import os
import paramiko
from scp import SCPClient



# scp -i ~/keys/gabe-master.pem /Users/gapaza/repos/gabe/hydra-chess/datasets/pt/millionsbase/millionsbase.zip ubuntu@3.145.44.57:/home/ubuntu/hydra-chess/datasets/pt/millionsbase


key_location = '~/keys/gabe-master.pem'
remote_user = 'ubuntu'
remote_address = '3.145.44.57'
remote_location = '/home/ubuntu/hydra-chess/datasets/pt/millionsbase-pt2'
transfer_file = os.path.join(config.pt_datasets_dir, 'millionsbase-pt2', 'millionsbase-pt2.zip')




def main():
    command = 'scp ' + '-i ' + key_location + ' ' + transfer_file + ' ' + remote_user + '@' + remote_address + ':' + remote_location
    print('SCP Command:', command)



if __name__ == '__main__':
    main()