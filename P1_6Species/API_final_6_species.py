'''
Author: Jennifer Sailor
Date: 4/13
Order in Sequence: 2

This is an API connection to GBIS
that finds the Butterflies/Lepidoptera in SA

ways to update: have it loop through taxon and saving
'''
import pygbif
import pandas as pd

#_________CHOOSE TAXON___________________________________________________________
# Set the taxon key for Danaus chrysippus
taxon_key = 7642610
# Set the taxon key for Cynthia cardui / Vanessa cardui
taxon_key = 4299368
# Set the taxon key for Belenois aurota
taxon_key = 1919181
# Set the taxon key for Bicyclus safitza/ Mycalesis rhacotis
taxon_key = 1893562
# Set the taxon key for Cacyreus marshalli
taxon_key = 1932752
# Set the taxon key for Eronia cleodora
taxon_key = 1920815


#_____________PARAMETERS_______________________________________________________
# Set the country to search for occurrences
continent_key = 'AFRICA'
# Set the number of results to retrieve
lim = 300
# Set the parameters for the GBIF occurrence search
params = {'taxonKey': taxon_key, 'continent': continent_key, 'limit': lim}
# Search for occurrences
occ_Lepidoptera = pygbif.occurrences.search(**params)
# Create a DataFrame from the results
df = pd.DataFrame.from_records(occ_Lepidoptera['results'])

#____________TEST______________________________________________________________
# Select only the relevant columns
df2 = df[['species', 'decimalLatitude', 'decimalLongitude']]
print(df2.head(10))

#__________COLLECT ALL____________________________________________________________
offset = lim
while len(occ_Lepidoptera['results']) == lim:
    params = {'taxonKey': taxon_key, 'continent': continent_key, 'limit': lim, 'offset': offset}
    occ_Lepidoptera = pygbif.occurrences.search(**params)
    df = df.append(pd.DataFrame.from_records(occ_Lepidoptera['results']))
    offset += lim

print(len(df))
df2 = df[['species', 'decimalLatitude', 'decimalLongitude']]
#__________CHOOSE CSV TO SAVE TO _______________________________________________
df.to_csv('Africa_Lepidoptera_Danaus_chrysippus.csv', index=False)
df.to_csv('Africa_Lepidoptera_Vanessa_cardui.csv', index=False)
df2.to_csv('Africa_Lepidoptera_Belenois_aurota_LL.csv', index=False)
df2.to_csv('Africa_Lepidoptera_Mycalesis_rhacotis_LL.csv', index=False)
df2.to_csv('Africa_Lepidoptera_Cacyreus_marshalli_LL.csv', index=False)
df2.to_csv('Africa_Lepidoptera_Eronia_cleodora_LL.csv', index=False)