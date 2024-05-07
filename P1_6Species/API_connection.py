'''
Author: Jennifer Sailor
Date: 4/13
Order in Sequence: 1

This is an API connection to GBIS
that finds the Butterflies/Lepidoptera in SA
'''
import pygbif
import pandas as pd

# Set the taxon key for Lepidoptera
taxon_key = 797

# Set the country to search for occurrences
#country = 'South Africa'
country_key = 'ZA'

# Set the number of results to retrieve
#According to website should have about 271,000 lines
lim = 300

# Set the parameters for the GBIF occurrence search
params = {'taxonKey': taxon_key, 'country': country_key, 'limit': lim}

# Search for occurrences of Lepidoptera in South Africa
occ_Lepidoptera = pygbif.occurrences.search(**params)

# Create a DataFrame from the results
df = pd.DataFrame.from_records(occ_Lepidoptera['results'])

# Select only the relevant columns
df2 = df[['species', 'decimalLatitude', 'decimalLongitude']]


print(df2.head(10))


offset = limit +1
while len(response['results']) == limit:
    params = {'taxonKey': taxon_key, 'country': country_key, 'limit': lim, 'offset' = offset}
    response = pygbif.occurrences.search(**params)
    df = df.append(pd.DataFrame.from_records(response['results']))
    offset += limit +1


#df.to_csv('South-Africa_Lepidoptera.csv', index=False)