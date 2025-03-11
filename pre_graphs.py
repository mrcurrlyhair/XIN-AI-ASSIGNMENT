import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# load cleaned dataset
clean_traffic_data = pd.read_csv('cleaned_traffic_accidents.csv')

# menu for graph selection
while True:
    print('\nSelect a graph to display')
    print('1 Distribution of vehicles Involved in Accidents')
    print('2 Accident Frequency by Hour / three or more vehicles')
    print('3 Accident Frequency by Day of the Week / three or more vehicles')
    print('4 Weather Condition vs. Multi-Vehicle Accidents / three or more vehicles')
    print('5 Correlation Heatmap')
    print('6 Quit')
    
    option = input('Enter your choice')

    high_risk = clean_traffic_data[clean_traffic_data['cars_involved'] >= 3]

        
    if option == '1':
        # grouping by 'cars_involved' and amount of accidnts
        accidents = clean_traffic_data['cars_involved'].value_counts().sort_index()
        fatal = clean_traffic_data.groupby('cars_involved')['injuries_fatal'].sum()
        fatal_percentage = (fatal/accidents * 100)

        # overlaping data on bar chart
        plt.figure(figsize=(8, 5))
        percentage = accidents.plot(kind='bar', color='blue', edgecolor='black', alpha=0.5, label='Total Accidents')
        fatal.plot(kind='bar', color='red', edgecolor='black', alpha=0.7, label='Fatal Accidents')
        
        # add percentages top the bar chart 
        for i, (total, fatal_count, fatal_percentage) in enumerate(zip(accidents, fatal, fatal_percentage)):
            if total :
                percentage.text(i, fatal_count + 1, f'{fatal_percentage:.1f}%', ha='center', fontsize=10, color='black')


        plt.xlabel('Number of Vehicles Involved')
        plt.ylabel('Count')
        plt.title('Total Accidents vs Fatal Accidents by Number of Vehicles Involved')
        plt.legend()
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

            # filter for weather 
            high_risk_weather = clean_traffic_data[clean_traffic_data['cars_involved'] >= 3][weather_columns].sum()
            
            # total accidents
            total_accidents = clean_traffic_data[weather_columns].sum()

            # high risk weather accidents
            high_risk_weather = high_risk[weather_columns].sum()

            # normalise data
            high_risk_percentage = (high_risk_weather / total_accidents) * 100


            # plot the data
            plt.figure(figsize=(12, 5))
            high_risk_percentage.plot(kind='bar', color='b')
            plt.xlabel('Weather Condition')
            plt.ylabel('Number of Multi-Vehicle Accidents')
            plt.title('Weather Condition vs Accidents Involving 3 or More Vehicles')
            plt.xticks(rotation=90)
            plt.show()
            plt.close()
        
    elif option == '5':
            # correlation heatmap
            plt.figure(figsize=(30, 24))
            corr_matrix = clean_traffic_data.corr()
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
            plt.xticks(fontsize=4)
            plt.yticks(fontsize=4)
            plt.title('Correlation Heatmap')
            plt.show()
            plt.close()
        
    elif option == '6':
            print('Quiting')
            break
    else:
            print('Invalid option, quiting')
            break
    
