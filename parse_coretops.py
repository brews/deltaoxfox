# 2018-02-19

# Parse raw coretop data and output a single clean dataset.

import datetime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pylab as plt

tempannual_nc_path = 'https://data.nodc.noaa.gov/thredds/dodsC/woa/WOA13/DATAv2/temperature/netcdf/decav/1.00/woa13_decav_t00_01v2.nc'
d18osw_nc_path = './data_raw/LeGrande_Schmidt2006 v1p1 d18o.nc'
mgca_path = './data_raw/Mg_Sites.xlsx'
margo_path = './data_raw/MARGO_d18O_LH_CT.xls'
environ_nc_template = './data_parsed/spp_environments/{}.nc'
coretopout_csv_template = './data_parsed/coretops-{}.csv'


def latlon2xyz(lat, lon):
    latr = np.deg2rad(lat)
    lonr = np.deg2rad(lon)
    x = np.cos(latr) * np.cos(lonr)
    y = np.cos(latr) * np.sin(lonr)
    z = np.sin(latr)
    return x, y, z


def chord_distance(lat1, lon1, lat2, lon2):
    x1, y1, z1 = latlon2xyz(lat1, lon1)
    x2, y2, z2 = latlon2xyz(lat2, lon2)
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


mgca = pd.read_excel(mgca_path, sheet_name='all_spp', na_values=-999)

margo = pd.read_excel(margo_path, skiprows=2)
margo = margo.rename(columns={'Unnamed: 15': 'species',
                              'Unnamed: 20': 'comments',
                              'Core': 'corename',
                              'Average d18O (‰ PDB) ': 'd18oc',
                              'Latitude (decimal, from -90 to +90)': 'latitude',
                              'Longitude (decimal, from -180 to +180)': 'longitude'})

# Columns:
# corename, latitude, longitude, species, d18O

commoncore = set(margo.corename) & set(mgca.corename)
margo_unique = margo.query("corename not in {}".format(list(commoncore)))

coretops = pd.concat([mgca.loc[:, ('corename', 'latitude', 'longitude', 'species', 'd18oc')],
                    margo_unique.loc[:, ('corename', 'latitude', 'longitude', 'species', 'd18oc')]],
                    ignore_index=True)

species_dictionary = {'Bu': 'bulloides',
                      'bu': 'bulloides',
                      'Rw': 'ruberwhite',
                      'rw': 'ruberwhite',
                      'ruber_w': 'ruberwhite',
                      'Sc': None,  # Not sure what this is.
                      'sc': None,  # Not sure what this is.
                      'Rp': 'ruberpink',
                      'rp': 'ruberpink',
                      'ruber_p': 'ruberpink',
                      'Pl': 'pachydermasin',
                      'pl': 'pachydermasin',
                      'pachy_s': 'pachydermasin',
                      'Pr': 'pachyderma',
                      'pr': 'pachyderma',
                      'sacc': 'sacculifer',
                      }

species_translator = {'species': species_dictionary}
coretops.replace(to_replace=species_translator, inplace=True)

coretops.dropna(inplace=True)

target_spp = ['bulloides', 'pachyderma', 'pachydermasin', 
              'ruberwhite', 'ruberpink', 'sacculifer']
spp_mask = [spp in target_spp for spp in coretops.species]
coretops = coretops.loc[np.array(spp_mask), :]

# TODO(brews): Need to account for number of datapoints to do weighted average.
coretops = coretops.groupby(['corename', 'species']).mean().reset_index()


# Add environment data to df
coretops['temp'] = np.nan
coretops['temp_ann'] = np.nan
coretops['d18osw'] = np.nan

# TODO(brews): Check the distribution of distances for each variable to be 
#     sure we're not getting something crazy high.

for spp, df in coretops.groupby('species'):

    print('Starting {}'.format(spp))

    if spp is None:
        continue

    environ_nc_path = environ_nc_template.format(spp)

    # try:
    #     with xr.open_dataset(environ_nc_path, decode_times=False) as ds:
    #         stacked = (ds.t_mn.sel(time=6.0, drop=True).stack(latlon=('lat', 'lon'))
    #                      .dropna('latlon'))

    #         for idx, site_df in df.iterrows():
    #             corename = site_df['corename']
    #             d = chord_distance(stacked.lat.values, stacked.lon.values, 
    #                                site_df.latitude, site_df.longitude)
    #             min_idx = d.argmin()
    #             close_value = np.asscalar(stacked[min_idx])
    #             coretops.loc[coretops.corename == corename, 'temp'] = close_value

    # except OSError:
    #     print('No file {} - Continuing'.format(environ_nc_path))
    #     pass


    for idx, site_df in df.iterrows():

        corename = site_df['corename']

        with xr.open_dataset(tempannual_nc_path, decode_times=False) as ds_tann:
            stacked = (ds_tann.t_mn.sel(depth=0, time=6.0, drop=True)
                              .stack(latlon=('lat', 'lon'))
                              .dropna('latlon'))
            d = chord_distance(stacked.lat.values, stacked.lon.values, 
                               site_df.latitude, site_df.longitude)
            min_idx = d.argmin()
            close_value = np.asscalar(stacked[min_idx])
            coretops.loc[coretops.corename == corename, 'temp_ann'] = close_value

        with xr.open_dataset(d18osw_nc_path) as ds_d18osw:
            stacked = (ds_d18osw.d18o.sel(depth=0, drop=True)
                                .stack(latlon=('lat', 'lon'))
                                .dropna('latlon'))
            d = chord_distance(stacked.lat.values, stacked.lon.values, 
                               site_df.latitude, site_df.longitude)
            min_idx = d.argmin()
            close_value = np.asscalar(stacked[min_idx])
            coretops.loc[coretops.corename == corename, 'd18osw'] = close_value

coretops.to_csv(coretopout_csv_template.format(datetime.date.today()), index=False)
