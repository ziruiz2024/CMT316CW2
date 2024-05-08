import scipy.io
import os
import shutil

mat = scipy.io.loadmat('D:\\APML\\CMT316CW2\\lists\\test_list.mat')
print(type(mat))


def copy_file(source_path, dest_path, folderName, fileName):
    try:
        mainFolder = dest_path + "\\" + folderName
        sourceFile = source_path + "\\" + folderName + "\\" + fileName
        
        if os.path.exists(mainFolder) == False:
            os.mkdir(mainFolder)
            
        targetFolder = mainFolder + "\\" + fileName
        shutil.copy(sourceFile, targetFolder)
       # print(f"File copied from '{source_path}' to '{dest_path}' successfully.")
    except IOError as e:
        print(f"Unable to copy file. {e}")

def main():
    parent_file = "D:\\APML\\images"
    dest_path = "D:\\APML\\images_new"
    
    for file in mat['file_list']:
        file_str = str(file[0][0])
        arr = file_str.split('/')
        
        
        copy_file(parent_file, dest_path, arr[0], arr[1])
    
main()