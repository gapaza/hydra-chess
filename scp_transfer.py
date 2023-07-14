import config
import os
import paramiko
from scp import SCPClient



# scp -i ~/keys/gabe-master.pem /Users/gapaza/repos/gabe/hydra-chess/datasets/pt/millionsbase/millionsbase.zip ubuntu@3.145.44.57:/home/ubuntu/hydra-chess/datasets/pt/millionsbase


key_location = '~/keys/gabe-master.pem'
remote_user = 'ubuntu'
remote_address = '3.138.120.76'
file_name = 'megaset-pt3-64-30p-int16'
remote_location = '/home/ubuntu/hydra-chess/datasets/pt/' + file_name
transfer_file = os.path.join(config.pt_datasets_dir, file_name, file_name + '.zip')
# transfer_file = os.path.join(config.ft_datasets_dir, file_name, file_name + '.zip')




def main():
    command = 'scp ' + '-i ' + key_location + ' ' + transfer_file + ' ' + remote_user + '@' + remote_address + ':' + remote_location
    print('SCP Command:', command)



if __name__ == '__main__':
    main()