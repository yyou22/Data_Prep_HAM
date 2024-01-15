import csv

#total = 1100

file = open('summary.csv','w')
writer = csv.writer(file)

writer.writerow(['nat', 'adv'])

for i in range(0, 3):

	with open('./tabu_data/data_' + str(i) + '_all.csv') as file_obj:

		reader_obj = csv.reader(file_obj)

		correct_nat = 0

		correct_adv = 0

		for row in reader_obj:
			if row[2] == row[3]:

				if row[4] == '0':
					correct_nat += 1
				
				if row[4] == '1':
					correct_adv += 1

	writer.writerow([str(correct_nat), str(correct_adv)])

