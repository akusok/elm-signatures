from PIL import Image 
import glob, os, shutil
import pandas as pd

here_path = os.getcwd()
folder = 'GPDSS10000'
#folder = 'MCYTDB'

here2 = os.path.join(here_path,folder)

imgs_folders = [os.path.join(here2,imgdir) for imgdir in os.listdir(here2)]

for here in imgs_folders:

    for infile in glob.glob(here+"/*.JPG"):
        file, ext = os.path.splitext(infile)
        im = Image.open(infile)
        rgb_im = im.convert('RGB')
        rgb_im.save(file + ".png", "PNG")
    for infile in glob.glob(here+"/*.jpg"):
        file, ext = os.path.splitext(infile)
        im = Image.open(infile)
        rgb_im = im.convert('RGB')
        rgb_im.save(file + ".png", "PNG") 

    
    to_move = os.path.join(here,'jpgfiles')

    if not os.path.exists(to_move):
        os.makedirs(to_move)
    
    for file in glob.glob(here+"/*.jpg"):
        shutil.move(file,to_move)
    
    for file in glob.glob(here+"/*.jpeg"):
        shutil.move(file,to_move)

if __name__ == "__main__":
    print("Here!")