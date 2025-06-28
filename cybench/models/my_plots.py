import csv
from collections import defaultdict

import matplotlib.pyplot as plt

# Replace with your CSV file path

crop_calendar_maize_US_file = 'cybench-data/maize/US/crop_calendar_maize_US.csv'
fpar_maize_US_file = 'cybench-data/maize/US/fpar_maize_US.csv'
meteo_maize_US_file = 'cybench-data/maize/US/meteo_maize_US.csv'
ndvi_maize_US_file = 'cybench-data/maize/US/ndvi_maize_US.csv'
soil_maize_US_file = 'cybench-data/maize/US/soil_maize_US.csv'
soil_moisture__maize_US_file  = 'cybench-data/maize/US/soil_moisture_maize_US.csv'
yield_maize_US_file = 'cybench-data/maize/US/yield_maize_US.csv'

def plot_crop_calendar(csv_file):
    x = []
    z = []
    y = []

    with open(csv_file, 'r', newline='') as file:
        print("opened")
        reader = csv.DictReader(file)
        for row in reader:
            y.append(float(row['sos']))  
            z.append(float(row['eos'])) 
            x.append(row['adm_id'])

        sos_dict = defaultdict(list)
        eos_dict = defaultdict(list)

        for i in range(len(x)):
            sos_dict[x[i]].append(y[i])
            eos_dict[x[i]].append(z[i])

        adm_ids = sorted(sos_dict.keys())
        avg_sos = [sum(sos_dict[adm_id]) / len(sos_dict[adm_id]) for adm_id in adm_ids]
        avg_eos = [sum(eos_dict[adm_id]) / len(eos_dict[adm_id]) for adm_id in adm_ids]

        plt.subplot(1, 2, 1)
        plt.plot(adm_ids, avg_sos, marker='o')
        plt.xlabel('adm_id')
        plt.ylabel('Average sos')
        plt.title('Average sos for each adm_id')

        plt.subplot(1, 2, 2)
        plt.plot(adm_ids, avg_eos, marker='o', color='orange')
        plt.xlabel('adm_id')
        plt.ylabel('Average eos')
        plt.title('Average eos for each adm_id')

        plt.tight_layout()
        plt.show()

def plot_fpar_ndvi(csv_file):
    with open(csv_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        x = []
        y = []
        z = []

        #md = {}
        md2 = {}
        for row in reader:
            if row['adm_id'] not in md2:
                #md[row['adm_id']] = []
                md2[row['adm_id']] = []
            #md[row['adm_id']].append(float(row['fpar']))
            md2[row['adm_id']].append(float(row['ndvi']))

        for location in md2:
            x.append(location)
            #y.append(sum(md[location]) / len(md[location]))
            z.append(sum(md2[location]) / len(md2[location]))

        # print(x[:10])
        # print(y[:10])
        # plt.plot(x, y, marker='o')
        # plt.xlabel('adm_id')
        # plt.ylabel('avg fpar')
        # plt.title('fpar for each adm_id')
        # plt.xticks(rotation=90)
        # plt.tight_layout()
        # plt.show()

        # plt.plot(x, z, marker='o')
        # plt.xlabel('adm_id')
        # plt.ylabel('avg ndvi')
        # plt.title('ndvi for each adm_id')
        # plt.xticks(rotation=90)
        # plt.tight_layout()
        # plt.show()

def plot_meteo_maize(csv_file):
    with open(csv_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        x = []
        y = []

        md = {}
        target_var = 'et0' # Change this to the desired variable: tmax,tmin,prec,rad,tavg,et0,cwb
        for row in reader:
            if row['adm_id'] not in md:
                md[row['adm_id']] = []
            md[row['adm_id']].append(float(row[target_var]))

        for location in md:
            x.append(location)
            y.append(sum(md[location]) / len(md[location]))

        # print(x[:10])
        # print(y[:10])
        # plt.plot(x, y, marker='o')
        # plt.xlabel('adm_id')
        # plt.ylabel('avg ' + target_var)
        # plt.title('Average ' + target_var + ' for each adm_id')
        # plt.xticks(rotation=90)
        # plt.tight_layout()
        # plt.show()

# plot_crop_calendar(crop_calendar_maize_US_file)
# plot_fpar_ndvi(ndvi_maize_US_file)
# plot_meteo_maize(meteo_maize_US_file)


