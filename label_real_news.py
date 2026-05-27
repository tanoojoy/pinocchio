import pandas as pd

#input and output file paths
input_file = "real_news.csv"
output_file = "real_news_with_labels.csv"

#read the CSV
df = pd.read_csv(input_file)

#add the new column 'label' with value 'real'
df["label"] = "real"

#save to a new CSV
df.to_csv(output_file, index=False)

print("Done. New file saved as:", output_file)