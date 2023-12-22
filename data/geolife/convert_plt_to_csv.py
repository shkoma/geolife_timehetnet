# file 삭제
# find . -name '*.csv' -exec rm {} \;
import os

def getUserId(id):
    val = ""
    if id < 10:
        val += "00"
        val += str(id)
    elif id < 100:
        val += "0"
        val += str(id)
    else:
        val = str(id)
    return val

total_user = 182
for id in range(total_user):
    user_id = getUserId(id)
    path = './Data/' + str(user_id) + '/Trajectory/'
    csv_folder = './Data/' + str(user_id) + '/csv/'
    plt_list = os.listdir(path)
    plt_list = [plt for plt in plt_list if plt.endswith('.plt')]

    if os.path.isdir(csv_folder) == False:
        os.mkdir(csv_folder)
        
    csv_folder
    csv_file = str(csv_folder) + str(user_id) + '.csv'
    columns = 'latitude,longitude,what,altitude,days,date,time\n'
    
    if os.path.isfile(csv_file) == False:
        with open(csv_file, 'w') as c_file:
            c_file.write(columns)  
            for plt_file in plt_list:
                if os.path.isfile(path + plt_file):
                    with open(path + plt_file, 'r') as p_file:
                        lines = p_file.readlines()
                        for line in lines[6:]:
                            c_file.write(line)