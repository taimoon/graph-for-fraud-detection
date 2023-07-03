from venv import create
from os.path import abspath
from os import getcwd,makedirs,listdir
from subprocess import run

for dir in ('data','results','results/history', 'save_results', 'save_results/visualization'):
    makedirs(dir,exist_ok=True)
    
# dir = ".env"
# if dir not in listdir():
#     print(f"create .env at {getcwd()}")
#     create(dir, with_pip=True)

# # where requirements.txt is in same dir as this script
# run(["bin/pip", "install", "-r", abspath("req.txt")], cwd=dir)
