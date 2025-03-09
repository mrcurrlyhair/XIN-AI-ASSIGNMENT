import pandas as pd

# load dataset
traffic_data = pd.read_csv('traffic_accidents.csv')

# rename num_units to cars_involved
traffic_data.rename(columns={'num_units': 'cars_involved'}, inplace=True)

# drop unnecessary columns
drop_columns = ['alignment', 'crash_date', 'crash_type', 'first_crash_type', 'damage', 'intersection_related_i']
traffic_data.drop(columns=drop_columns, inplace=True)

# remove rows containing unknown
toremove = []
for index, row in traffic_data.iterrows():
    if "UNKNOWN" in row.astype(str).values:  
        toremove.append(index)
traffic_data = traffic_data.drop(toremove)

# convert relevant columns to integer
traffic_data['crash_day_of_week'] = traffic_data['crash_day_of_week'].astype(int)
traffic_data['cars_involved'] = traffic_data['cars_involved'].astype(int)
traffic_data['crash_hour'] = traffic_data['crash_hour'].astype(int)

traffic_data['injuries_fatal'] = traffic_data['injuries_fatal'] > 0

# OHE variables
OHEcolumns = ['weather_condition', 'lighting_condition', 'roadway_surface_cond', 'traffic_control_device', 'trafficway_type', 'road_defect', 'prim_contributory_cause', 'most_severe_injury']
traffic_data = pd.get_dummies(traffic_data, columns=OHEcolumns)

# save cleaned data
traffic_data.to_csv('cleaned_traffic_accidents.csv', index=False)

print('data cleaned')


D