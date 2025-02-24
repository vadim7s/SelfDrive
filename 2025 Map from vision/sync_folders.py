"""
Synchronizes two folders by deleting extra files in either of them.
Files that exist in one folder but not in the other will be removed.
used to remove extra images of maps so we have only matching pairs 
left for training a model
"""
import os



folder1 = 'C://SelfDrive//2025 Map from vision//img'
folder2 =  'C://SelfDrive//2025 Map from vision//map_img'
folder3 =  'C://SelfDrive//2025 Map from vision//sem_img'


# Get sets of filenames in both folders
files1 = set(os.listdir(folder1))
files2 = set(os.listdir(folder2))
files3 = set(os.listdir(folder3))

# delete empty files (except maps)
for file in files1:
    file_path = os.path.join(folder1, file)
    if os.path.getsize(file_path)< 4000:
        os.remove(file_path)
        print(f"Deleted: {file_path}")

for file in files3:
    file_path = os.path.join(folder3, file)
    if os.path.getsize(file_path)< 4000:
        os.remove(file_path)
        print(f"Deleted: {file_path}")

# Get sets of filenames left in both folders
files1 = set(os.listdir(folder1))
files2 = set(os.listdir(folder2))
files3 = set(os.listdir(folder3))



# Identify extra files in each folder
extra_in_folder1 = files1 - files2
extra_in_folder2 = files2 - files1
extra_in_folder3 = files3 - files1 
# Delete extra files in folder1
for file in extra_in_folder1:
    file_path = os.path.join(folder1, file)
    os.remove(file_path)
    #print(f"Deleted: {file_path}")

# Delete extra files in folder2
for file in extra_in_folder2:
    file_path = os.path.join(folder2, file)
    os.remove(file_path)
    #print(f"Deleted: {file_path}")

# Delete extra files in folder2
for file in extra_in_folder3:
    file_path = os.path.join(folder3, file)
    os.remove(file_path)
    #print(f"Deleted: {file_path}")

print("Synchronization complete.")