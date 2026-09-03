"""Subsets a CARRA NetCDF file to the southern Spitsbergen study area.

Longitude values are normalized where required, a spatial index window is identified
from latitude and longitude coordinates, and the subset is written as compressed NetCDF."""
from pathlib import Path
import numpy as np
import xarray as xr
input_nc = Path('D:\\CARRA\\OPAD\\dd17df2fe7c7878c1914df636cb13c62.nc')
output_nc = input_nc.with_name(input_nc.stem + '_south_spitsbergen.nc')
lon_min = 11.0
lon_max = 21.0
lat_min = 76.0
lat_max = 78.0
print('Otwieranie pliku:')
print(input_nc)
try:
    ds = xr.open_dataset(input_nc, decode_times=True)
except Exception as exc:
    print('\nNie udało się otworzyć pliku NetCDF.')
    print('Błąd:', exc)
    print('\nJeżeli w komunikacie widzisz brak silnika netcdf4, zainstaluj pakiet poleceniem:')
    print('C:\\Users\\Kuba\\AppData\\Local\\Programs\\Python\\Python39\\python.exe -m pip install netCDF4')
    print('\nPo instalacji uruchom skrypt ponownie.')
    raise
print('\nWymiary pliku wejściowego:')
print(ds.sizes)
print('\nZmienne:')
print(list(ds.data_vars))
possible_lat_names = ['latitude', 'lat']
possible_lon_names = ['longitude', 'lon']
lat_name = next((name for name in possible_lat_names if name in ds.variables), None)
lon_name = next((name for name in possible_lon_names if name in ds.variables), None)
if lat_name is None or lon_name is None:
    raise KeyError(f'Nie znaleziono zmiennych latitude/longitude.\nDostępne zmienne: {list(ds.variables)}')
lat = ds[lat_name]
lon = ds[lon_name]
print('\nZmienna szerokości geograficznej:', lat_name)
print('Wymiary:', lat.dims)
print('\nZmienna długości geograficznej:', lon_name)
print('Wymiary:', lon.dims)
lon_normalized = (lon + 180.0) % 360.0 - 180.0
mask = (lat >= lat_min) & (lat <= lat_max) & (lon_normalized >= lon_min) & (lon_normalized <= lon_max)
mask_values = np.asarray(mask.values, dtype=bool)
if not mask_values.any():
    raise ValueError('W podanym zakresie nie znaleziono żadnych komórek CARRA.\nSprawdź lon_min/lon_max/lat_min/lat_max.')
if lat.ndim != 2:
    raise ValueError(f'Oczekiwano 2D latitude, otrzymano {lat.ndim}D o wymiarach {lat.dims}.')
y_dim, x_dim = lat.dims
rows, cols = np.where(mask_values)
y_min_idx = int(rows.min())
y_max_idx = int(rows.max())
x_min_idx = int(cols.min())
x_max_idx = int(cols.max())
print('\nWyznaczone indeksy:')
print(f'{y_dim}: {y_min_idx} – {y_max_idx}')
print(f'{x_dim}: {x_min_idx} – {x_max_idx}')
subset = ds.isel({y_dim: slice(y_min_idx, y_max_idx + 1), x_dim: slice(x_min_idx, x_max_idx + 1)})
print('\nWymiary po przycięciu:')
print(subset.sizes)
subset_lat = subset[lat_name]
subset_lon = (subset[lon_name] + 180.0) % 360.0 - 180.0
print('\nRzeczywisty zasięg przyciętego rastra:')
print('Latitude:', float(subset_lat.min()), '–', float(subset_lat.max()))
print('Longitude:', float(subset_lon.min()), '–', float(subset_lon.max()))
encoding = {}
for name, var in subset.data_vars.items():
    if var.ndim >= 2:
        encoding[name] = {'zlib': True, 'complevel': 4, 'shuffle': True}
print('\nZapisywanie:')
print(output_nc)
subset.to_netcdf(output_nc, engine='netcdf4', format='NETCDF4', encoding=encoding)
ds.close()
print('\n============================================')
print('GOTOWE')
print('============================================')
print('Plik wyjściowy:')
print(output_nc)
