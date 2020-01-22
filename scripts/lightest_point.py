import cv2

def draw_circle(image, x, y):
	cv2.circle(image, (x, y), 5, (0,255,0), -1)
	cv2.imwrite("new.png", image)


def get_coordinates(output_file):
	with open(output_file, 'r') as f:
		content = f.readlines()
	content = [x.strip() for x in content]

	counter = 0
	for n in content:
		if counter == 0:
			coordinate1 = n.split()
			x_coordinate1 = int(coordinate1[0])
			y_coordinate1 = int(coordinate1[1]) 
			new_coordinate1 = (str(x_coordinate1) + " " + str(y_coordinate1))
		
		if counter == 16:
			coordinate2 = n.split()
			x_coordinate2 = int(coordinate2[0])
			y_coordinate2 = int(coordinate2[1])
			new_coordinate2 = (str(x_coordinate2) + " " + str(y_coordinate2))
		counter += 1

	return new_coordinate1, new_coordinate2;


# find lightest coordinate in 20 x 20 pixel
def lightest_point(filename, output_file, coordinate, flag):

	image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

	coordinate = coordinate.split(" ")
	x_coordinate = int(coordinate[0])
	y_coordinate = int(coordinate[1])

	lightest = image[x_coordinate, y_coordinate]
	lightest_x = 0
	lightest_y = 0	
		

	offset = 20
	if flag == "l":
		offset = 20
	elif flag == "r":
		offset = -20

	for i in range(x_coordinate, x_coordinate + 20):
		for j in range(y_coordinate, y_coordinate + 20):
			pixel = image[i, j]
#			print(i, j)
			if pixel > lightest:
				lightest = pixel
				lightest_x = i
				lightest_y = j
	# draw_circle(image, lightest_x, lightest_y)	

	print(lightest_x)
	print(lightest_y)

	return lightest_x, lightest_y;


'''
def lightest_point(filename, output_file):

	image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

	lightest = image[0, 0]
	lightest_x = 0
	lightest_y = 0

	for i in range(0,image.shape[0]):
		for j in range(0, image.shape[1]):
			pixel = image[i, j]
#			print(i, j)
			if pixel > lightest:
				lightest = pixel
				lightest_x = i
				lightest_y = j
	# draw_circle(image, lightest_x, lightest_y)	

	print(lightest)
	print(lightest_x)
	print(lightest_y)

	return lightest_x, lightest_y;
'''

if __name__== '__main__':
	filename = '0a.png'
	output_file = '0a.png.txt'

	left_coordinate, right_coordinate = get_coordinates(output_file)
	print("left coordinate")
#	print(left_coordinate)
	print("right coordinate")
#	print(right_coordinate)
	x, y = lightest_point(filename, output_file, left_coordinate, "l")
	x, y = lightest_point(filename, output_file, right_coordinate, "r")


