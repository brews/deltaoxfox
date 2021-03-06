import pkgutil
import io
import pandas as pd


RESOURCE_STR = 'sststats/foram_sst_stats.csv'


def get_csv_resource(fl, **kwargs):
    """Read CSV package resource, output for Pandas
    """
    bytesin = io.BytesIO(pkgutil.get_data('deltaoxfox', fl))
    data = pd.read_csv(bytesin, **kwargs)
    return data


SSTSTATS_DF = get_csv_resource(RESOURCE_STR, index_col='foramtype')


def foram_sst_minmax(foram):
    """Get SST (°C) min,max for coretop planktic foram records"""
    # For legacy DA, we need to normalize species names.
    if str(foram) in ['T. sacculifer', 'G. sacculifer']:
        foram = 'T. sacculifer'
    elif str(foram) in ['G. ruber pink', 'G. ruber white', 'G. ruber']:
        foram = 'G. ruber'
    elif str(foram) in ['N. pachyderma sinistral', 'N. pachyderma']:
        foram = 'N. pachyderma'

    try:
        target_stats = SSTSTATS_DF.loc[foram].copy()
    except KeyError:
        avail = list(SSTSTATS_DF.index)
        raise KeyError('`foram` arg {} not found in available forams: {}'.format(foram, avail))

    return target_stats['min'], target_stats['max']
