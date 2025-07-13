import pandas as pd
import matplotlib.pyplot as plt
import os

csv_path = r'cybench\data\maize\NL\meteo_lai_bgr.csv'
df = pd.read_csv(csv_path, parse_dates=['date'])

#df: Dataframe, feature: "lai/bgr", title of the plot, ylable (feature), filename to save the plot

def plot_feature_over_time(df, feature, title, ylabel, filename):
    plt.figure(figsize=(12, 6))
    for (adm_id, year), group in df.groupby(['adm_id', 'year']):
        plt.plot(group['date'], group[feature], label=f'{adm_id} {year}')
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("model_results", filename))
    plt.close()

def plot_feature_per_year(df, feature, title, ylabel, filename, admin_id, year):
    filtered_df = df[((df['adm_id'] == admin_id) & (df['year'] == year)) | ((df['adm_id'] == admin_id) & (df['year'] == year+1))]
    #filtered_df = df[(df['adm_id'] == admin_id) & (df['year'] == year)]
    #filtered_df = df[df['adm_id'] == admin_id]
    print(filtered_df.shape)
    # Save the output of filtered_df[feature] to a text file
    output_path = os.path.join("model_results", f"{filename}_feature_values.txt")
    filtered_df[feature].to_string(open(output_path, "w"))
    plt.figure(figsize=(12, 6))
    color = "blue"
    if feature == "lai":
        color = "green"
    plt.plot(filtered_df['date'], filtered_df[feature], label=f'{admin_id},{year}', color=color)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("model_results", filename))
    plt.close()

plot_feature_per_year(df, "lai", "Cum LAI over one season", "Cum LAI", "lai_year", "NL42", 2019)
#plot_feature_per_year(df, "bgr", "Biomass Growth Rate", "Biomass Accumulated", "bgr_year", "NL42", 2019)

