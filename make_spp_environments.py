#! /usr/bin/env python

# Create xarray.Datasets and netCDF files that contain the growth environment 
# climate variables for different foram species.

import os
import datetime

import numpy as np
import xarray as xr
import matplotlib.pylab as plt


#TODO(brews): Need weighted mean for months.
#TODO(brews): We don't need 'concat_dim=month' for any of these.

# Remember, '%02.f'%12 to stuff zero to len == 2.
daysinmonth=[0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
ncenv_path = './data_parsed/spp_environments/'
temp_nc_template = 'https://data.nodc.noaa.gov/thredds/dodsC/woa/WOA13/DATAv2/temperature/netcdf/decav/1.00/woa13_decav_t{}_01v2.nc'


with xr.open_dataset(temp_nc_template.format('00'), decode_times=False) as d:
    blank = xr.full_like(d.t_mn.sel(drop=True, depth=0), np.nan)


def make_pachy_s():
    out = blank.copy()
    out.squeeze()

    with xr.open_dataset(temp_nc_template.format('00'), decode_times=False) as d:
        # out[:] = d.t_mn.where(d.depth <= 100).mean(dim='depth')
        out[:] = d.t_mn.sel(depth = 0)

    nc_paths = [temp_nc_template.format(x) for x in ['07', '08', '09', '10']]
    with xr.open_mfdataset(nc_paths, decode_times=False, concat_dim='month') as d:
        # t_mn = d.t_mn.where(d.depth <= 100).mean(dim=['depth', 'month', 'time'])
        t_mn = d.t_mn.sel(depth = 0).mean(dim=['month', 'time'])
        out[dict(lat=np.abs(out.lat) < 60)] = t_mn[dict(lat=np.abs(t_mn.lat) < 60)]

    return out


def make_pachy_d():
    out = blank.copy()
    out.squeeze()

    nc_paths = [temp_nc_template.format(x) for x in ['06', '07', '08', '09']]
    with xr.open_mfdataset(nc_paths, decode_times=False, concat_dim='month') as d:
        # t_mn = d.t_mn.where(d.depth <= 100).mean(dim=['depth', 'month', 'time'])
        t_mn = d.t_mn.sel(depth = 0).mean(dim=['month', 'time'])
        out[:] = t_mn
    return out


def make_bulloides():
    out = blank.copy()
    out.squeeze()

    nc_paths = [temp_nc_template.format(x) for x in ['07', '08', '09', '10']]
    with xr.open_mfdataset(nc_paths, decode_times=False, concat_dim='month') as d:
        # t_mn = d.t_mn.where(d.depth <= 100).mean(dim=['depth', 'month', 'time'])
        t_mn = d.t_mn.sel(depth = 0).mean(dim=['month', 'time'])
        out[:] = t_mn

    nc_paths = [temp_nc_template.format(x) for x in ['12', '01', '02', '03', '04']]
    with xr.open_mfdataset(nc_paths, decode_times=False, concat_dim='month') as d:
        # t_mn = d.t_mn.where(d.depth <= 100).mean(dim=['depth', 'month', 'time'])
        t_mn = d.t_mn.sel(depth = 0).mean(dim=['month', 'time'])
        out[dict(lat=out.lat > 15)] = t_mn[dict(lat=t_mn.lat > 15)]


    nc_paths = [temp_nc_template.format(x) for x in ['07', '08', '09', '10']]
    with xr.open_mfdataset(nc_paths, decode_times=False, concat_dim='month') as d:
        # t_mn = d.t_mn.where(d.depth <= 100).mean(dim=['depth', 'month', 'time'])
        t_mn = d.t_mn.sel(depth = 0).mean(dim=['month', 'time'])
        out[dict(lat=out.lat < -20)] = t_mn[dict(lat=t_mn.lat < -20)]

    return out 


def make_sacculifer():
    return make_ruber_w(max_depth=100)


def make_ruber_w(max_depth=50):
    out = blank.copy()
    out.squeeze()

    with xr.open_dataset(temp_nc_template.format('00'), decode_times=False) as d:
        # out[:] = d.t_mn.where(d.depth <= max_depth).mean(dim='depth')
        out[:] = d.t_mn.sel(depth = 0)

    nc_paths = [temp_nc_template.format(x) for x in ['07', '08', '09', '10']]
    with xr.open_mfdataset(nc_paths, decode_times=False, concat_dim='month') as d:
        # t_mn = d.t_mn.where(d.depth <= max_depth).mean(dim=['depth', 'month', 'time'])
        t_mn = d.t_mn.sel(depth = 0).mean(dim=['month', 'time'])
        out[dict(lat=out.lat < -23)] = t_mn[dict(lat=t_mn.lat < -23)]

    nc_paths = [temp_nc_template.format(x) for x in ['12', '01', '02', '03', '04']]
    with xr.open_mfdataset(nc_paths, decode_times=False, concat_dim='month') as d:
        t_mn = d.t_mn.sel(depth = 0).mean(dim=['month', 'time'])
        out[dict(lat=out.lat > 23)] = t_mn[dict(lat=t_mn.lat > 23)]
    return out


def main():
    """Write netcdfs with spp environment information to directory"""
    os.makedirs(ncenv_path, exist_ok=True)
    make_ruber_w().to_netcdf(os.path.join(ncenv_path, 'ruber_w.nc'))
    make_sacculifer().to_netcdf(os.path.join(ncenv_path, 'sacculifer.nc'))
    make_bulloides().to_netcdf(os.path.join(ncenv_path, 'bulloides.nc'))
    make_pachy_d().to_netcdf(os.path.join(ncenv_path, 'pachy_d.nc'))
    make_pachy_s().to_netcdf(os.path.join(ncenv_path, 'pachy_s.nc'))


if __name__ == '__main__':
    main()