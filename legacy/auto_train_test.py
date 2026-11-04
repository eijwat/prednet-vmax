# time python3 auto_train_test.py

import subprocess
import sys

################################################
period="40000"
savemodel="10000"
this_file="auto_train_test.py"
channels="1,48,96,192"
init_model="init/56_1945_5000.pth"
num_train=20
image_size="160,120"
test_code="auto_benham.py"
################################################


################################################
train_images=["data_unity/rbg/train_list.txt","data_sim/000/train_list.txt","data_sim/001/train_list.txt","data_sim/010/train_list.txt","data_sim/011/train_list.txt","data_sim/100/train_list.txt","data_sim/101/train_list.txt","data_sim/110/train_list.txt","data_sim/111/train_list.txt","data_sim/555/train_list.txt"]
out_folder=["modelz/unity", "modelz/sim000","modelz/sim001","modelz/sim010","modelz/sim011","modelz/sim100","modelz/sim101","modelz/sim110","modelz/sim111","modelz/sim555"]
################################################

len_train_images=len(train_images)
len_out_folder=len(out_folder)

if len_train_images != len_out_folder:
	print("check list!")
	sys.exit()

for loop in range(len_train_images):

	res = subprocess.run(["mkdir", out_folder[loop]], stdout=subprocess.PIPE)
	sys.stdout.buffer.write(res.stdout)

	for i in range(num_train):
		
		#training
		res = subprocess.run(["python3","main.py","-i", train_images[loop], "--period", period, "--save", savemodel,"--lr_rate", "0.9", "--initmodel", init_model, "-s", image_size], stdout=subprocess.PIPE)
		sys.stdout.buffer.write(res.stdout)
		
		out0=out_folder[loop] + "/" + str(i)
		res = subprocess.run(["mkdir", out0], stdout=subprocess.PIPE)
		sys.stdout.buffer.write(res.stdout)

		out1=out0 + "/"
		out2=out1 + "log_t.txt"	
		out3=out1 + this_file
		
		res = subprocess.run(["mv", "models", out1], stdout=subprocess.PIPE)
		sys.stdout.buffer.write(res.stdout)
		
		res = subprocess.run(["mv", "log_t.txt", out2], stdout=subprocess.PIPE)
		sys.stdout.buffer.write(res.stdout)

		res = subprocess.run(["cp",this_file, out3], stdout=subprocess.PIPE)
		sys.stdout.buffer.write(res.stdout)
		
		res = subprocess.run(["mkdir", "models"], stdout=subprocess.PIPE)
		sys.stdout.buffer.write(res.stdout)

	#test	
	res = subprocess.run(["python3",test_code,"-m", out_folder[loop], "-n", str(num_train)], stdout=subprocess.PIPE)
	sys.stdout.buffer.write(res.stdout)
	


################################################
train_images=["data_iacc3/0.txt","data_iacc3/1.txt","data_iacc3/2.txt","data_iacc3/3.txt","data_iacc3/4.txt","data_iacc3/12.txt","data_iacc3/13.txt","data_iacc3/14.txt","data_iacc3/15.txt","data_iacc3/16.txt"]
out_folder=["modelz/iacc3_0", "modelz/iacc3_1","modelz/iacc3_2","modelz/iacc3_3","modelz/iacc3_4","modelz/iacc3_12","modelz/iacc3_13","modelz/iacc3_14","modelz/iacc3_15","modelz/iacc3_16"]
################################################
"""
################################################
train_images=["data_iacc3/14.txt"]
out_folder=["modelz/iacc3_14"]
################################################
"""
len_train_images=len(train_images)
len_out_folder=len(out_folder)

if len_train_images != len_out_folder:
	print("check list!")
	sys.exit()

for loop in range(len_train_images):

	res = subprocess.run(["mkdir", out_folder[loop]], stdout=subprocess.PIPE)
	sys.stdout.buffer.write(res.stdout)

	for i in range(num_train):
		
		#training
		res = subprocess.run(["python3","main.py","-seq", train_images[loop], "--period", period, "--save", savemodel,"--lr_rate", "0.9", "--initmodel", init_model, "-s", image_size], stdout=subprocess.PIPE)
		sys.stdout.buffer.write(res.stdout)
		
		out0=out_folder[loop] + "/" + str(i)
		res = subprocess.run(["mkdir", out0], stdout=subprocess.PIPE)
		sys.stdout.buffer.write(res.stdout)

		out1=out0 + "/"
		out2=out1 + "log_t.txt"	
		out3=out1 + this_file
		
		res = subprocess.run(["mv", "models", out1], stdout=subprocess.PIPE)
		sys.stdout.buffer.write(res.stdout)
		
		res = subprocess.run(["mv", "log_t.txt", out2], stdout=subprocess.PIPE)
		sys.stdout.buffer.write(res.stdout)

		res = subprocess.run(["cp",this_file, out3], stdout=subprocess.PIPE)
		sys.stdout.buffer.write(res.stdout)
		
		res = subprocess.run(["mkdir", "models"], stdout=subprocess.PIPE)
		sys.stdout.buffer.write(res.stdout)

	#test	
	res = subprocess.run(["python3",test_code,"-m", out_folder[loop], "-n", str(num_train)], stdout=subprocess.PIPE)
	sys.stdout.buffer.write(res.stdout)
