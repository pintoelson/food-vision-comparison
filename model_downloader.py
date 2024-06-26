# FOlder: https://drive.google.com/drive/folders/1g6RlMvEsPPnWMgxzrWjseyJZfOlqM0Ba?usp=drive_link
# https://drive.google.com/file/d/1ZsKl2ef8RvcsLVARyIF9cdXD11KrueS0/view?usp=drive_link

import gdown
import os

# url = 'https://drive.google.com/file/d/1ZsKl2ef8RvcsLVARyIF9cdXD11KrueS0/view?usp=drive_link'
# output = 'model_V0.pth'

# gdown.download(url, output, fuzzy = True)


def download_model(model):
    url = get_url(model)
    output = 'models/'
    if os.path.exists(output):    
        print('Models Folder Exists')
    else:
        os.mkdir('models')
    if url is None:
        print('Model not in gdrive')
        return None
    if os.path.exists(output + model +'.pth'):
        print('File already exists')
    else:
        gdown.download(url, output, fuzzy = True)

def get_url(model):
    if model == 'model_V0':
        return 'https://drive.google.com/file/d/1ZsKl2ef8RvcsLVARyIF9cdXD11KrueS0/view?usp=drive_link'
    elif model == 'model_V1':
        return 'https://drive.google.com/file/d/1sNlIGwJC8nmk-naXIgVlaMsy9QraXJ-8/view?usp=drive_link'
    elif model == 'model_V2':
        return 'https://drive.google.com/file/d/1fSxmp-TGouWBx997VyUG7jwalM2V7-Pr/view?usp=drive_link'
    elif model == 'model_V3':
        return 'https://drive.google.com/file/d/11oiw1_Ayk7V3XxenFFjVf12LkdiqCr0I/view?usp=drive_link'
    else:
        return None