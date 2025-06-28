def filter_by_adm_id(input_csv: str, output_csv: str, adm_id: str):
    """
    Reads the input CSV file, filters rows by the given adm_id,
    and writes the filtered rows (including header) to the output CSV file.
    """
    with open(input_csv, 'r', encoding='utf-8') as infile, open(output_csv, 'w', encoding='utf-8') as outfile:
        header = infile.readline()
        outfile.write(header)
        i = 0
        for line in infile:
            if line.strip() == "":
                continue
            fields = line.split(',')
            if input_csv.endswith('yield_maize_US.csv'):
                if len(fields) > 1 and fields[2].startswith(adm_id):
                    outfile.write(line)
            else:
                if len(fields) > 1 and fields[1].startswith(adm_id):
                    outfile.write(line)

files = [
    r"cybench\data\maize\US\crop_calendar_maize_US.csv",
    r"cybench\data\maize\US\fpar_maize_US.csv",
    r"cybench\data\maize\US\meteo_maize_US.csv",
    r"cybench\data\maize\US\ndvi_maize_US.csv",
    r"cybench\data\maize\US\soil_maize_US.csv",
    r"cybench\data\maize\US\soil_moisture_maize_US.csv",
    r"cybench\data\maize\US\yield_maize_US.csv"
]

for file in files:
    input_file = file
    output_file = file.replace('US', 'Indiana')
    print("output_file", output_file)
    filter_by_adm_id(input_file, output_file, 'US-18')

#filter_by_adm_id(r"cybench\data\maize\US\crop_calendar_maize_US.csv",r"cybench\data\maize\Indiana\crop_calendar_maize_Indiana.csv", 'US-18')

