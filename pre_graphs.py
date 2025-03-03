import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load cleaned data
clean_traffic_data = pd.read_csv('cleaned_traffic_accidents.csv')

# distribution of number of cars involved
plt.figure(figsize=(8, 5))
plt.hist(clean_traffic_data['cars_involved'], bins=10, edgecolor='black')
plt.xlabel('Number of Cars Involved')
plt.ylabel('Count')
plt.title('Distribution of Cars Involved in Accidents')
plt.show()
plt.close()

# accident frequency by hour
plt.figure(figsize=(10,5))
clean_traffic_data['crash_hour'].value_counts().sort_index().plot(kind='bar', color='b')
plt.xlabel('hour of the day')
plt.ylabel('number of accidents')
plt.title('accident frequency by hour')
plt.show()
plt.close()

# accident frequency by day of the week
plt.figure(figsize=(10,5))
clean_traffic_data['crash_day_of_week'].value_counts().sort_index().plot(kind='bar', color='b')
plt.xlabel('day of the week (0=monday, 6=sunday)')
plt.ylabel('number of accidents')
plt.title('accident frequency by day of the week')
plt.show()
plt.close()

# weather condition v multi-vehicle accidents 
weather_columns = [
    'weather_condition_BLOWING SAND, SOIL, DIRT',
    'weather_condition_BLOWING SNOW',
    'weather_condition_CLEAR',
    'weather_condition_CLOUDY/OVERCAST',
    'weather_condition_FOG/SMOKE/HAZE',
    'weather_condition_FREEZING RAIN/DRIZZLE',
    'weather_condition_OTHER',
    'weather_condition_RAIN',
    'weather_condition_SEVERE CROSS WIND GATE',
    'weather_condition_SLEET/HAIL',
    'weather_condition_SNOW'
]

# sum of multi-vehicle accidents per weather condition
multi_vehicle_counts = clean_traffic_data[clean_traffic_data['cars_involved'] >= 3][weather_columns].sum()

# plot the data
plt.figure(figsize=(12,5))
multi_vehicle_counts.plot(kind='bar', color='b')
plt.xlabel('weather condition')
plt.ylabel('number of multi-vehicle accidents')
plt.title('weather condition vs. multi-vehicle accidents')
plt.xticks(rotation=45)
plt.show()
plt.close()

# correlation heatmap NEEDS FIXING
plt.figure(figsize=(12,8))
corr_matrix = clean_traffic_data.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.title('correlation heatmap')
plt.show()
plt.close()










