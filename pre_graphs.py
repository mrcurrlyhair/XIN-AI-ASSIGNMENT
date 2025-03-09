import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# load cleaned dataset
clean_traffic_data = pd.read_csv('cleaned_traffic_accidents.csv')

# menu for graph selection
while True:
    print('\nSelect a graph to display')
    print('1 Distribution of Cars Involved in Accidents')
    print('2 Accident Frequency by Hour / three or more vehicles')
    print('3 Accident Frequency by Day of the Week / three or more vehicles')
    print('4 Weather Condition vs. Multi-Vehicle Accidents / three or more vehicles')
    print('5 Correlation Heatmap')
    print('6 Quit')
    
    option = input('Enter your choice')

    high_risk = clean_traffic_data[clean_traffic_data['cars_involved'] >= 3]

        
    if option == '1':
            # distribution of number of cars involved
            plt.figure(figsize=(8, 5))
            plt.hist(clean_traffic_data['cars_involved'], bins=10, edgecolor='black')
            plt.xlabel('Number of Cars Involved')
            plt.ylabel('Count')
            plt.title('Distribution of Cars Involved in Accidents')
            plt.show()
            plt.close()
        
    elif option == '2':
            # accident frequency by hour / three or more vehicles
            plt.figure(figsize=(10, 5))
            high_risk['crash_hour'].value_counts().sort_index().plot(kind='bar', color='b')
            plt.xlabel('Hour of the Day')
            plt.ylabel('Number of High-Risk Accidents')
            plt.title('Accident Frequency by Hour (3 or more)')
            plt.show()
            plt.close()
        
    elif option == '3':
            # accident frequency by day of the week / three or more vehicles
            plt.figure(figsize=(10, 5))
            high_risk['crash_day_of_week'].value_counts().sort_index().plot(kind='bar', color='b')
            plt.xlabel('Day of the Week (0=Monday, 6=Sunday)')
            plt.ylabel('Number of High-Risk Accidents')
            plt.title('Accident Frequency by Day of the Week (3 or more)')
            plt.show()
            plt.close()
        
    elif option == '4':
            # weather condition vs multi-vehicle accidents / three or more vehicles
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

            # filter for wether 
            high_risk_weather = clean_traffic_data[clean_traffic_data['cars_involved'] >= 3][weather_columns].sum()
            
            # plot the data
            plt.figure(figsize=(12, 5))
            high_risk_weather.plot(kind='bar', color='b')
            plt.xlabel('Weather Condition')
            plt.ylabel('Number of Multi-Vehicle Accidents')
            plt.title('Weather Condition vs. Multi-Vehicle Accidents')
            plt.xticks(rotation=90)
            plt.show()
            plt.close()
        
    elif option == '5':
            # correlation heatmap
            plt.figure(figsize=(12, 8))
            corr_matrix = clean_traffic_data.corr()
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.title('Correlation Heatmap')
            plt.show()
            plt.close()
        
    elif option == '6':
            print('Quiting')
            break
    else:
            print('Invalid option, quiting')
            break
    
