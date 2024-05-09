# Open the file for reading

def load_confusion_file(file_path):
    arr = []
    with open(file_path, 'r') as file:
        # Read the file line by line
        for line in file:
            # Process each line (e.g., print it)
            line = line.rstrip("\n")
            arr.append(line.split(" "))
    return arr

def load_misclassification_file(file_path):
    arr = []
    with open(file_path, 'r') as file:
        # Read the file line by line
        for line in file:
            # Process each line (e.g., print it)
            line = line.rstrip("\n")
            arr.append(line.split("  "))
    return arr

def dump (lst, file_path):
    with open(file_path, 'a') as file:
        for item in lst:
            file.write(item[0] + "  " + item[1] + "  " + str(item[2]) + "  " + str(item[3]) + "  " + str(item[4]) + '\n')

def merge(cm, mm):
    merged_arr = []
    for e in cm:
        if e[0] in [ins[0] for ins in mm]:
            index = 0
            for i, row in enumerate(mm):
                if row[0] == e[0]:
                    index = i
                    break
            merged_arr.append([e[0], e[1], e[2], mm[i][1], mm[i][2]])
            continue
        
        merged_arr.append([e[0], e[1], e[2], "None", "None"])
    return merged_arr


dump(merge(load_confusion_file("D:\\APML\\Result_Archive\\confusion_matrix.txt"),
      load_misclassification_file("D:\\APML\\Result_Archive\\misclassification_report.txt")
      ), "D:\\APML\\test_folder\\final_report.txt")