# This script will go through the result of the confusion matrix, and find the most common points of failure for every class.

# Specify the file path
file_path = "D:\\APML\\Result_Archive\\false_tests.txt"


def get_classifications():
    cl_err = []
    
    with open(file_path, 'r') as file:
        
        for line in file:
            res = line.split("  ")
            res[2] = res[2].strip()
            
            cl_err.append([res[1], res[2]])
        
    
    return cl_err


def get_category_list(cat, err_arr):
    return list(filter(lambda instance: instance[0] == cat, err_arr)) 


def get_maximum_error(cat_list):
    arr = []
        
    for r in cat_list:
        # print(arr)
        if r[1] in [ins[1] for ins in arr]:
            index = 0
            for i, row in enumerate(arr):
                if row[1] == r[1]:
                    index = i
                    break
                
            arr[index][2] = arr[index][2] + 1
        else:
            arr.append([r[0], r[1], 1])
    
    max = 0
    index = 0
    
    # print(arr)
    
    for i in range(0, len(arr)):
        if arr[i][2] > max:
            index = i
            max = arr[i][2]
    
    return arr[index]
        
def dump (lst, file_path):
    with open(file_path, 'a') as file:
        for item in lst:
            file.write(item[0] + "  " + item[1] + "  " + str(item[2]) + '\n')
    

def generate_report(err_arr):
    visited = []
    arr = []

    for cl in err_arr:
        if cl[0] not in visited:
            arr.append(get_maximum_error(
                get_category_list(cl[0], err_arr)
            ))
            visited.append(cl[0])    
    
    dump(arr, "D:\\APML\\test_folder\\misclassification_report.txt")
    

generate_report(get_classifications())
        