import csv

complaint_results_path = './inf_output_complaint.csv'
request_results_path = './inf_output_request_test_20230630.csv'

results = [["filename", "complaint", "request"]]

comlaint_dict = ['no', 'yes']

request_dict = ['affil', 'presta']

with open(complaint_results_path, 'r') as complaint_results:
    for line in complaint_results.readlines():
        line = line.split(',')
        results.append([line[0].split('/')[-1], comlaint_dict[int(line[2])], ''])

with open(request_results_path, 'r') as request_results:
    cnt = 1
    for line in request_results.readlines():
        line = line.split(',')
        results[cnt][2] = request_dict[int(line[2])]
        cnt += 1

f = open('inf_results_comlaint_and_request_devel.csv', 'w')
csv_writer = csv.writer(f)
csv_writer.writerows(results)
    
