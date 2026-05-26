import os, shutil, random 
frames_dir = r'C:\Users\apps\Downloads\Week4\Dataset\images\frames' 
train_dir = r'C:\Users\apps\Downloads\Week4\Dataset\images\train' 
val_dir = r'C:\Users\apps\Downloads\Week4\Dataset\images\val' 
test_dir = r'C:\Users\apps\Downloads\Week4\Dataset\images\test' 
frames = sorted(os.listdir(frames_dir)) 
random.seed(42) 
step_train = len(frames) // 100 
train = [frames[i*step_train] for i in range(100)] 
remaining = [f for f in frames if f not in train] 
step_val = len(remaining) // 40 
val = [remaining[i*step_val] for i in range(40)] 
test = [f for f in remaining if f not in val] 
for f in train: shutil.copy(os.path.join(frames_dir, f), os.path.join(train_dir, f)) 
for f in val: shutil.copy(os.path.join(frames_dir, f), os.path.join(val_dir, f)) 
for f in test: shutil.copy(os.path.join(frames_dir, f), os.path.join(test_dir, f)) 
print('Train:', len(train)) 
print('Val:', len(val)) 
print('Test:', len(test)) 
