import os

path = '/home/data/MyBackUP/Projects/ELM/02_Signatures/MCYTDB'

folders = [os.path.join(path,imgdir) for imgdir in os.listdir(path)]

restart_list = []
windows = [32,64,128, 256]

move_small = False
restart = False

for elm in folders:
    print(elm)
