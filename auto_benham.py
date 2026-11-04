
# python3 auto_benham.py

"""
モデルの位置を指定すると、
ベンハムのテストを行う。

python3 auto_benham.py -m modelz/test -n 20
"""

import subprocess
import sys
import argparse

parser = argparse.ArgumentParser(description='auto_benham.py')
parser.add_argument('--modelz_path', '-m', default='modelz/test', help='Path to modelz_path')
parser.add_argument('--number_folders', '-n', default=1, type=int, help='number of folders')
args = parser.parse_args()


############################
path_home=args.modelz_path
num_folders=args.number_folders
test_path=["bht", "bht_rev", "bht_static","bhtr","bhtr_rev","bhtr_static"]
this_file="auto_benham.py"
num_models=4
numx=10000
len_test_path=len(test_path)
image_size="160,120"
channel_size="3,48,96,192"
path_sum=path_home+"/sum"
path_jpg="test_19y_0.jpg"
###################################

o=path_home + "/" + this_file
res = subprocess.run(["cp", this_file, o], stdout=subprocess.PIPE)
sys.stdout.buffer.write(res.stdout)	

res = subprocess.run(["mkdir", path_sum], stdout=subprocess.PIPE)
sys.stdout.buffer.write(res.stdout)

#test
for i in range(num_folders):
	
	path_folder=path_home + "/" + str(i)
	path_model=path_folder + "/models"

	for j in range(len_test_path):
		
		path_out_result=path_folder + "/r_" + test_path[j]
		test_images="test_images/" + test_path[j] + "_test_list.txt"

		res = subprocess.run(["mkdir", path_out_result], stdout=subprocess.PIPE)
		sys.stdout.buffer.write(res.stdout)
		
		for k in range(num_models):
			
			num=(k+1)*numx
			
			model_pth=path_model + "/" + str(num) + ".pth"
					
			path_r=path_out_result + "/result_" + str(num)
			path_log=path_r + "/log_p.txt"
								
			res = subprocess.run(["python3","main.py","--test", "--initmodel", model_pth,"-i", test_images, "--channels", channel_size, "-s", image_size], stdout=subprocess.PIPE)
									
			res = subprocess.run(["mv", "result", path_r], stdout=subprocess.PIPE)
			sys.stdout.buffer.write(res.stdout)
			
			res = subprocess.run(["mv", "log_p.txt", path_log], stdout=subprocess.PIPE)
			sys.stdout.buffer.write(res.stdout)
				
			res = subprocess.run(["mkdir", "result"], stdout=subprocess.PIPE)
			sys.stdout.buffer.write(res.stdout)

#sum
for i in range(num_folders):
	
	path_folder=path_home + "/" + str(i)
		
	for j in range(len_test_path):
												
		path_read=path_folder + "/" + "r_" + test_path[j]

		for k in range(num_models):
			
			num=(k+1)*numx
			
			r=path_read+"/result_" + str(num) + "/"+ path_jpg
			o=path_sum + "/" + str(i) + "_" + test_path[j] + "_" + str(num) + "_" + path_jpg
			
			res = subprocess.run(["cp", r, o], stdout=subprocess.PIPE)
			sys.stdout.buffer.write(res.stdout)	
