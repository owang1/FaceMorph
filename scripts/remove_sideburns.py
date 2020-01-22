# Shifts 1st and 17th dlib points inward to remove sideburns

'''
def lightest_point():

  return x_offset, y_offset; 

'''

def remove_sideburns(numStart, numEnd, outputFolder, newOutputFolder):

	print("remove sideburns")
	for num in range(numStart, numEnd):
		print(num)
		# Read output file into list
		
		try:
			output_file1 = outputFolder + str(num) + "a.png.txt"
			output_file2 = outputFolder + str(num) + "b.png.txt"

			new_output_file1 = newOutputFolder + str(num) + "a_new.png.txt" 
			new_output_file2 = newOutputFolder + str(num) + "b_new.png.txt"

			with open(output_file1, 'r') as f1:
				content1 = f1.readlines()
			with open(output_file2, 'r') as f2:
				content2 = f2.readlines()

			content1 = [x.strip() for x in content1]
			content2 = [x.strip() for x in content2]
		   
			counter = 0
			for n1, n2  in zip(content1, content2):
				if counter == 0: 
				# Shift 1st landmark point down & right 10 pixels        
					coordinate1 = n1.split()
					x_coordinate1 = int(coordinate1[0]) + 10
					y_coordinate1 = int(coordinate1[1]) + 10
					new_coordinate1 = (str(x_coordinate1) + " " +  str(y_coordinate1))       
					content1[counter] = new_coordinate1

					coordinate2 = n2.split()
					x_coordinate2 = int(coordinate2[0]) + 10
					y_coordinate2 = int(coordinate2[1]) + 10
					new_coordinate2 = (str(x_coordinate2) + " " + str(y_coordinate2))
					content2[counter] = new_coordinates2

				if counter == 16:
				# Shifts 17th landmark point down & left 10 pixels
					coordinate1 = n1.split()
					x_coordinate1 = int(coordinate1[0]) - 20
					y_coordinate1 = int(coordinate1[1]) + 20
					new_coordinate1 = (str(x_coordinate1) + " " + str(y_coordinate1))
					content1[counter] = new_coordinate1

					coordinate2 = n2.split()
					x_coordinate2 = int(coordinate2[0]) - 20
					y_coordinate2 = int(coordinate2[1]) + 20
					new_coordinate2 = (str(x_coordinate2) + " " + str(y_coordinate2))
					content2[counter] = new_coordinates2

				counter += 1 


		    # Write content to new file
			with open(new_output_file1, 'w') as f1:
				for item in content1:
					f1.write(item + '\n')

			with open(new_output_file2, 'w') as f2:
				for item in content2:
					f2.write(item + '\n')


		
		except Exception:
			print(Exception)
			pass
		

