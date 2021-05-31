counter = 0

# Retrieve types of Interventions
import os
from os import listdir
from os.path import isfile, join
import json
import csv

cto_dir = '/mnt/nas2/data/systematicReview/clinical_trials_gov/XMLs/'
intervention_writefile = '/mnt/nas2/data/systematicReview/clinical_trials_gov/intervention_data2.csv'
missed_writefile = '/mnt/nas2/data/systematicReview/clinical_trials_gov/missed_data.csv'

def inter_iterator(full_filename, d):
  
    if 'intervention_type' in d:
        intervention_type = d['intervention_type']
        intervention_type = ' '.join(intervention_type.split())
    else:
        intervention_type = 'N.A.'

    if 'intervention_name' in d:
        intervention_name = d['intervention_name']
        intervention_name = ' '.join(intervention_name.split())
    else:
        intervention_name = 'N.A.'
        
    if 'description' in d:
        intervention_description = d['description']
        intervention_description = ' '.join(intervention_description.split())
    else:
        intervention_description = 'N.A.'

    if 'arm_group_label' in d:
        intervention_arm_group_label = d['arm_group_label']
        intervention_arm_group_label = ' '.join(intervention_arm_group_label.split())
    else:
        intervention_arm_group_label = 'N.A.'
        
    if 'other_name' in d:
        intervention_other_name = d['other_name']
        intervention_other_name = ' '.join(intervention_other_name.split())
    else:
        intervention_other_name = 'N.A.'
        
    resultInt = [str(full_filename).replace("\n", ""), str(intervention_type).replace("\n", ""), str(intervention_name).replace("\n", ""), str(intervention_description).replace("\n", ""), str(intervention_arm_group_label).replace("\n", ""), str(intervention_other_name).replace("\n", "")]
    return resultInt
    

directory = os.fsencode(cto_dir)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    full_filename = cto_dir + filename
    
    #if counter <= 100:
    with open(full_filename, 'r') as f, open(intervention_writefile, 'a+', newline='\n') as i_wf, open(missed_writefile, 'a+', newline='\n') as m_wf:
        writer = csv.writer(i_wf, delimiter='\t')
        m_writer = csv.writer(m_wf, delimiter='\t')

        d = json.load(f)
        try:
            intervention = d['clinical_study']['intervention']

            # If instance is a dictionary
            if isinstance(intervention, dict):
                intervention_write = inter_iterator(full_filename, intervention)
                writer.writerow(intervention_write)
                #print(counter)

            # If instance is a list
            elif isinstance(intervention,list):
                for eachEle in intervention:
                    intervention_write = inter_iterator(full_filename, eachEle)
                    writer.writerow(intervention_write)
                    #print(counter)


            counter = counter + 1

        except:
            #print("No intervnetion found from the record (%s)" % traceback.format_exc())
            print(full_filename)
            #print(counter)
            #m_writer.writerow([str(filename)])
            counter = counter + 1      
    #else:
    #    break