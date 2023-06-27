import config
import os
import paramiko
from scp import SCPClient



# scp -i ~/keys/gabe-master.pem /Users/gapaza/repos/gabe/hydra-chess/datasets/pt/millionsbase/millionsbase.zip ubuntu@3.145.44.57:/home/ubuntu/hydra-chess/datasets/pt/millionsbase


key_location = '~/keys/gabe-master.pem'
remote_user = 'ubuntu'
remote_address = '3.145.44.57'
remote_location = '/home/ubuntu/hydra-chess/datasets/ft/lc0_standard'
transfer_file = os.path.join(config.ft_datasets_dir, 'lc0_standard', 'lc0_standard.zip')




def main():
    command = 'scp ' + '-i ' + key_location + ' ' + transfer_file + ' ' + remote_user + '@' + remote_address + ':' + remote_location
    print('SCP Command:', command)



if __name__ == '__main__':
    main()