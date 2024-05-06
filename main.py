import os
os.system("pip install -r requirements.txt")
os.system("python train.py")
print("Model trained. Now running model.py")
os.system("python model.py")