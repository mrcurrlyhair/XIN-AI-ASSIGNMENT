import pandas as pd
import matplotlib.pyplot as plt


# load cleaned data
clean_traffic_data = pd.read_csv('cleaned_traffic_accidents.csv')

# Distribution of number of cars involved
plt.figure(figsize=(8, 5))
plt.hist(clean_traffic_data['cars_involved'], bins=10, edgecolor='black')
plt.xlabel('Number of Cars Involved')
plt.ylabel('Count')
plt.title('Distribution of Cars Involved in Accidents')
plt.show()
plt.close()









