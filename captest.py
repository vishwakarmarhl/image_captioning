import shutil
lines = []
with open("train_images.csv") as file:
    for line in file:
        line = line.strip() #or some other preprocessing
        lines.append(line) #storing everything in memory!
        shutil.copy2(line, './out/')