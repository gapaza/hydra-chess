import config
import os
import paramiko
from scp import SCPClient



# scp -i ~/keys/gabe-master.pem /Users/gapaza/repos/gabe/hydra-chess/datasets/pt/millionsbase/millionsbase.zip ubuntu@3.145.44.57:/home/ubuntu/hydra-chess/datasets/pt/millionsbase


key_location = '~/keys/gabe-master.pem'
remote_user = 'ubuntu'
remote_address = '3.138.120.76'
file_name = 'lc0_standard_small_128'
run_type = 'dc'

remote_location = '/home/ubuntu/hydra-chess/datasets/'+run_type+'/' + file_name

if run_type == 'pt':
    transfer_file = os.path.join(config.pt_datasets_dir, file_name, file_name + '.zip')
elif run_type == 'ft':
    transfer_file = os.path.join(config.ft_datasets_dir, file_name, file_name + '.zip')
elif run_type == 'dc':
    transfer_file = os.path.join(config.dc_datasets_dir, file_name, file_name + '.zip')



def main():
    command = 'scp ' + '-i ' + key_location + ' ' + transfer_file + ' ' + remote_user + '@' + remote_address + ':' + remote_location
    print('SCP Command:', command)



if __name__ == '__main__':
    main()