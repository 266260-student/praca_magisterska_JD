"""Extracts glacier surface velocity along a profile and at points representing three equal sections.

The profile is divided into thirds and point locations are set at L/6, L/2 and 5L/6.
The script combines the 2018-2022 GeoTIFF products with 2022-2025 ITS_LIVE NetCDF products."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from shapely.ops import linemerge
root_dir = Path('D:\\Studia\\praca_magisterska\\predkosci_svalbard_2018-2022')
raster_dir = root_dir / 'predkosci_svalbard'
netcdf_dir = root_dir / 'ITS_LIVE_path14'
vector_dir = root_dir / 'wektory'
output_dir = root_dir / 'wyniki_2018-2025_path14'
glacier_name = 'Scheelebreen'
profile_path = vector_dir / f'{glacier_name}_profile.gpkg'
outline_path = vector_dir / f'{glacier_name}_outline.gpkg'
point_paths = {'lower': vector_dir / f'{glacier_name}_point.gpkg', 'middle': vector_dir / f'{glacier_name}_point2.gpkg', 'upper': vector_dir / f'{glacier_name}_point3.gpkg'}
point_labels = {'lower': 'P1 — 1/6 L', 'middle': 'P2 — 1/2 L', 'upper': 'P3 — 5/6 L'}
basemap_path = vector_dir / 'mapa_bazowa_esri_800.tif'
spacing = 120
(output_dir / 'csv').mkdir(parents=True, exist_ok=True)
(output_dir / 'figures').mkdir(parents=True, exist_ok=True)

def prepare_profile_geometry(geom):
    """Zamienia MultiLineString na LineString; wybiera najdłuższą część, jeśli trzeba."""
    if geom.geom_type == 'LineString':
        return geom
    if geom.geom_type == 'MultiLineString':
        merged = linemerge(geom)
        if merged.geom_type == 'LineString':
            return merged
        if merged.geom_type == 'MultiLineString':
            print('UWAGA: profil składa się z kilku oddzielnych części.')
            print('Do analizy wybrano najdłuższy fragment.')
            return max(list(merged.geoms), key=lambda line: line.length)
    raise ValueError(f'Nieobsługiwany typ geometrii profilu: {geom.geom_type}')

def points_along_line(line, spacing_m):
    """Tworzy punkty co spacing_m wzdłuż profilu; interpolowana jest tylko geometria."""
    distances = np.arange(0, line.length, spacing_m)
    if len(distances) == 0 or distances[-1] < line.length:
        distances = np.append(distances, line.length)
    points = [line.interpolate(distance) for distance in distances]
    return (distances, points)

def nan_percent(array):
    array = np.asarray(array)
    if array.size == 0:
        return np.nan
    return round(np.isnan(array).sum() / array.size * 100, 2)

def read_basemap(path):
    """Wczytuje mapę bazową GeoTIFF bez reprojekcji."""
    if not path.exists():
        raise FileNotFoundError(f'Nie znaleziono mapy bazowej:\n{path}')
    with rasterio.open(path) as src:
        bounds = src.bounds
        crs = src.crs
        count = src.count
        print('\nMapa bazowa:')
        print('Ścieżka:', path)
        print('CRS zapisany w pliku:', crs)
        print('Liczba kanałów:', count)
        print('Zasięg:', bounds)
        if count >= 3:
            image = src.read([1, 2, 3])
            image = np.moveaxis(image, 0, -1)
            if image.dtype != np.uint8:
                image = image.astype(np.float32)
                finite = np.isfinite(image)
                if np.any(finite):
                    low = np.nanpercentile(image[finite], 1)
                    high = np.nanpercentile(image[finite], 99)
                    if high > low:
                        image = (image - low) / (high - low)
                        image = np.clip(image, 0, 1)
        else:
            image = src.read(1, masked=True)
    return (image, bounds, crs)

def extract_dates_from_filename(path):
    name = path.name
    match = re.search('(\\d{8})-(\\d{8})', name)
    if not match:
        raise ValueError(f'Nie znaleziono zakresu dat w nazwie pliku: {name}')
    date_start = pd.to_datetime(match.group(1), format='%Y%m%d')
    date_end = pd.to_datetime(match.group(2), format='%Y%m%d')
    date_mid = date_start + (date_end - date_start) / 2
    return (date_start, date_end, date_mid)

def sample_velocity_magnitude(src, coords):
    """
    Kanał 1 = Vx, kanał 2 = Vy.
    V = sqrt(Vx^2 + Vy^2). Bez interpolacji i ekstrapolacji.
    """
    if src.count < 2:
        raise ValueError(f'Raster {src.name} ma tylko {src.count} kanał(y). Oczekiwano co najmniej 2.')
    values = []
    samples = src.sample(coords, indexes=[1, 2], masked=True)
    for sample in samples:
        vx, vy = (sample[0], sample[1])
        if np.ma.is_masked(vx) or np.ma.is_masked(vy):
            values.append(np.nan)
            continue
        try:
            vx = float(vx)
            vy = float(vy)
        except (TypeError, ValueError):
            values.append(np.nan)
            continue
        if not np.isfinite(vx) or not np.isfinite(vy):
            values.append(np.nan)
            continue
        values.append(np.hypot(vx, vy))
    return np.array(values, dtype=float)

def extract_dates_itslive(path):
    name = path.stem
    parts = name.split('_X_')
    if len(parts) == 2:
        match_1 = re.search('(\\d{8}T\\d{6})', parts[0])
        match_2 = re.search('(\\d{8}T\\d{6})', parts[1])
        if match_1 is not None and match_2 is not None:
            date_start = pd.to_datetime(match_1.group(1), format='%Y%m%dT%H%M%S')
            date_end = pd.to_datetime(match_2.group(1), format='%Y%m%dT%H%M%S')
            if date_end < date_start:
                date_start, date_end = (date_end, date_start)
            date_mid = date_start + (date_end - date_start) / 2
            return (date_start, date_end, date_mid)
    timestamps = re.findall('\\d{8}T\\d{6}', name)
    if len(timestamps) < 2:
        raise ValueError(f'Nie udało się odczytać dwóch dat z pliku:\n{path.name}')
    unique_dates = []
    for timestamp in timestamps:
        dt = pd.to_datetime(timestamp, format='%Y%m%dT%H%M%S')
        if len(unique_dates) == 0 or dt.date() != unique_dates[-1].date():
            unique_dates.append(dt)
    if len(unique_dates) < 2:
        raise ValueError(f'Nie udało się jednoznacznie rozpoznać pary dat w:\n{path.name}')
    date_start, date_end = (unique_dates[0], unique_dates[1])
    if date_end < date_start:
        date_start, date_end = (date_end, date_start)
    date_mid = date_start + (date_end - date_start) / 2
    return (date_start, date_end, date_mid)

def find_velocity_subdataset(netcdf_path):
    with rasterio.open(netcdf_path) as dataset:
        subdatasets = dataset.subdatasets
    if len(subdatasets) == 0:
        raise ValueError(f'Plik nie zawiera subdatasetów:\n{netcdf_path}')
    for subdataset in subdatasets:
        variable_name = subdataset.rsplit(':', 1)[-1].strip('"')
        if variable_name == 'v':
            return subdataset
    raise ValueError(f"Nie znaleziono zmiennej 'v' w pliku:\n{netcdf_path}")

def sample_velocity_v(src, coords):
    """Pobiera v = velocity magnitude [m/year]. Braki pozostają NaN."""
    values = []
    samples = src.sample(coords, indexes=1, masked=True)
    for sample in samples:
        value = sample[0]
        if np.ma.is_masked(value):
            values.append(np.nan)
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            values.append(np.nan)
            continue
        if not np.isfinite(value) or value < 0:
            values.append(np.nan)
            continue
        values.append(value)
    return np.array(values, dtype=float)
rasters = sorted(list(raster_dir.rglob('*.tif')) + list(raster_dir.rglob('*.tiff')))
print('\n====================================')
print('DANE 2018–2022')
print('====================================')
print('\nZnaleziono plików TIFF:', len(rasters))
if len(rasters) == 0:
    raise FileNotFoundError(f'Nie znaleziono rastrów w folderze:\n{raster_dir}')
raster_records = []
for raster_path in rasters:
    date_start, date_end, date_mid = extract_dates_from_filename(raster_path)
    raster_records.append({'path': raster_path, 'date_start': date_start, 'date_end': date_end, 'date': date_mid, 'source': 'GeoTIFF 2018-2022'})
raster_dates = pd.DataFrame(raster_records).sort_values('date').reset_index(drop=True)
print('Zakres czasowy TIFF:')
print('Od:', raster_dates['date_start'].min())
print('Do:', raster_dates['date_end'].max())
netcdf_files = sorted(netcdf_dir.rglob('*.nc'))
print('\n====================================')
print('DANE ITS_LIVE 2022–2025')
print('====================================')
print('\nZnaleziono plików NetCDF:', len(netcdf_files))
if len(netcdf_files) == 0:
    raise FileNotFoundError(f'Nie znaleziono plików .nc w:\n{netcdf_dir}')
records = []
for path in netcdf_files:
    try:
        date_start, date_end, date_mid = extract_dates_itslive(path)
        v_subdataset = find_velocity_subdataset(path)
        records.append({'path': path, 'v_subdataset': v_subdataset, 'date_start': date_start, 'date_end': date_end, 'date': date_mid, 'source': 'ITS_LIVE 2022-2025'})
    except Exception as error:
        print('\nPominięto plik:', path.name)
        print('Powód:', error)
if len(records) == 0:
    raise ValueError('Nie znaleziono żadnych poprawnych produktów ITS_LIVE.')
velocity_dates = pd.DataFrame(records).sort_values('date').reset_index(drop=True)
print('Poprawnych produktów ITS_LIVE:', len(velocity_dates))
print('Zakres czasowy ITS_LIVE:')
print('Od:', velocity_dates['date_start'].min())
print('Do:', velocity_dates['date_end'].max())
profile_gdf = gpd.read_file(profile_path)
outline_gdf = gpd.read_file(outline_path)
point_gdfs = {name: gpd.read_file(path) for name, path in point_paths.items()}
if profile_gdf.empty:
    raise ValueError('Profil nie zawiera geometrii.')
if outline_gdf.empty:
    raise ValueError('Obrys lodowca nie zawiera geometrii.')
for name, gdf in point_gdfs.items():
    if gdf.empty:
        raise ValueError(f"Warstwa punktowa '{name}' nie zawiera geometrii.")
if profile_gdf.crs is None:
    profile_gdf = profile_gdf.set_crs('EPSG:3413', allow_override=True)
if outline_gdf.crs is None:
    outline_gdf = outline_gdf.set_crs('EPSG:3413', allow_override=True)
for name in point_gdfs:
    if point_gdfs[name].crs is None:
        point_gdfs[name] = point_gdfs[name].set_crs('EPSG:3413', allow_override=True)
print('\nCRS warstw:')
print('Profil:', profile_gdf.crs)
print('Obrys:', outline_gdf.crs)
for name, gdf in point_gdfs.items():
    print(f'Punkt {name}:', gdf.crs)
basemap_img, basemap_bounds, basemap_crs = read_basemap(basemap_path)
profile_geom = prepare_profile_geometry(profile_gdf.geometry.iloc[0])
outline_geom = outline_gdf.geometry.iloc[0]
profile_start = profile_geom.interpolate(0)
profile_end = profile_geom.interpolate(profile_geom.length)
profile_length = float(profile_geom.length)
print('\nDługość profilu [m]:', round(profile_length, 2))
third = profile_length / 3.0
target_distances = {'lower': profile_length / 6.0, 'middle': profile_length / 2.0, 'upper': 5.0 * profile_length / 6.0}
interval_bounds = {'lower': (0.0, third), 'middle': (third, 2.0 * third), 'upper': (2.0 * third, profile_length)}
point_geoms = {name: profile_geom.interpolate(distance) for name, distance in target_distances.items()}
print('\n====================================')
print('NOWE POŁOŻENIE 3 PUNKTÓW')
print('====================================')
print('Profil podzielono na trzy równe odcinki.')
print('Punkty ustawiono w środkach tych odcinków.')
for name, new_geom in point_geoms.items():
    gdf = point_gdfs[name].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:3413', allow_override=True)
    elif str(gdf.crs).upper() != 'EPSG:3413':
        gdf = gdf.to_crs('EPSG:3413')
    gdf.at[gdf.index[0], 'geometry'] = new_geom
    point_gdfs[name] = gdf
    gdf.to_file(point_paths[name], driver='GPKG')
    print(f'Zaktualizowano {point_paths[name].name}: X={new_geom.x:.3f}, Y={new_geom.y:.3f}')
point_metadata = []
for name, geom in point_geoms.items():
    distance_along = profile_geom.project(geom)
    distance_to_profile = geom.distance(profile_geom)
    interval_start, interval_end = interval_bounds[name]
    print(f'\n{name.upper()}:')
    print('Etykieta:', point_labels[name])
    print('Współrzędne:', geom.x, geom.y)
    print('Początek przedziału [m]:', round(interval_start, 2))
    print('Koniec przedziału [m]:', round(interval_end, 2))
    print('Środek przedziału / odległość wzdłuż profilu [m]:', round(distance_along, 2))
    print('Pozycja względna [% długości]:', round(distance_along / profile_length * 100.0, 2))
    print('Odległość od profilu [m]:', round(distance_to_profile, 6))
    point_metadata.append({'point': name, 'label': point_labels[name], 'x': geom.x, 'y': geom.y, 'interval_start_m': interval_start, 'interval_end_m': interval_end, 'distance_along_profile_m': distance_along, 'relative_position_percent': distance_along / profile_length * 100.0, 'distance_to_profile_m': distance_to_profile})
point_metadata_df = pd.DataFrame(point_metadata)
point_metadata_csv = output_dir / 'csv' / f'{glacier_name}_points_equal_sections_metadata.csv'
point_metadata_df.to_csv(point_metadata_csv, index=False)
print('\nZapisano metadane nowych punktów:')
print(point_metadata_csv)
with rasterio.open(raster_dates.iloc[0]['path']) as src:
    print('\n====================================')
    print('PIERWSZY RASTER 2018–2022')
    print('====================================')
    print('CRS:', src.crs)
    print('Rozdzielczość:', src.res)
    print('Liczba kanałów:', src.count)
    print('Typ danych:', src.dtypes)
    print('NoData:', src.nodatavals)
    if src.count < 2:
        raise ValueError('Raster ma mniej niż dwa kanały. Nie można obliczyć modułu prędkości.')
first_v_subdataset = velocity_dates.iloc[0]['v_subdataset']
with rasterio.open(first_v_subdataset) as src:
    print('\n====================================')
    print('PIERWSZY SUBDATASET v ITS_LIVE')
    print('====================================')
    print('CRS:', src.crs)
    print('Rozdzielczość:', src.res)
    print('Liczba kanałów:', src.count)
    print('NoData:', src.nodata)
    print('Typ danych:', src.dtypes)
    print('Zasięg:', src.bounds)
distances, sample_points = points_along_line(profile_geom, spacing_m=spacing)
coords_profile = [(point.x, point.y) for point in sample_points]
coords_points = {name: [(geom.x, geom.y)] for name, geom in point_geoms.items()}
print('\nPróbkowanie profilu:')
print('Odstęp:', spacing, 'm')
print('Liczba punktów:', len(coords_profile))
print('Długość:', round(distances[-1], 2), 'm')
profile_matrix_old = []
point_series_old = {name: [] for name in point_paths}
print('\n====================================')
print('EKSTRAKCJA TIFF 2018–2022')
print('====================================')
for i, row in raster_dates.iterrows():
    raster_path = row['path']
    with rasterio.open(raster_path) as src:
        values_profile = sample_velocity_magnitude(src, coords_profile)
        profile_matrix_old.append(values_profile)
        for name in point_paths:
            value = sample_velocity_magnitude(src, coords_points[name])[0]
            point_series_old[name].append(value)
    if (i + 1) % 25 == 0 or i + 1 == len(raster_dates):
        print(f'Przetworzono {i + 1}/{len(raster_dates)} rastrów TIFF')
profile_matrix_old = np.array(profile_matrix_old, dtype=float)
point_series_old = {name: np.array(values, dtype=float) for name, values in point_series_old.items()}
profile_matrix_new = []
point_series_new = {name: [] for name in point_paths}
print('\n====================================')
print('EKSTRAKCJA ITS_LIVE 2022–2025')
print('====================================')
for i, row in velocity_dates.iterrows():
    v_subdataset = row['v_subdataset']
    with rasterio.open(v_subdataset) as src:
        profile_values = sample_velocity_v(src, coords_profile)
        profile_matrix_new.append(profile_values)
        for name in point_paths:
            value = sample_velocity_v(src, coords_points[name])[0]
            point_series_new[name].append(value)
    if (i + 1) % 25 == 0 or i + 1 == len(velocity_dates):
        print(f'Przetworzono {i + 1}/{len(velocity_dates)} produktów ITS_LIVE')
profile_matrix_new = np.array(profile_matrix_new, dtype=float)
point_series_new = {name: np.array(values, dtype=float) for name, values in point_series_new.items()}
if profile_matrix_old.shape[1] != profile_matrix_new.shape[1]:
    raise ValueError(f'Macierze starego i nowego zbioru mają różną liczbę punktów profilu:\nTIFF: {profile_matrix_old.shape}\nITS_LIVE: {profile_matrix_new.shape}')
combined_dates = pd.concat([raster_dates[['date_start', 'date_end', 'date', 'source']], velocity_dates[['date_start', 'date_end', 'date', 'source']]], ignore_index=True)
combined_profile_matrix = np.vstack([profile_matrix_old, profile_matrix_new])
combined_point_series = {name: np.concatenate([point_series_old[name], point_series_new[name]]) for name in point_paths}
sort_order = np.argsort(combined_dates['date'].to_numpy())
combined_dates = combined_dates.iloc[sort_order].reset_index(drop=True)
combined_profile_matrix = combined_profile_matrix[sort_order, :]
combined_point_series = {name: values[sort_order] for name, values in combined_point_series.items()}
print('\n====================================')
print('POŁĄCZONE DANE 2018–2025')
print('====================================')
print('Liczba TIFF 2018–2022:', len(raster_dates))
print('Liczba ITS_LIVE 2022–2025:', len(velocity_dates))
print('Łączna liczba produktów:', len(combined_dates))
print('Zakres czasowy:', combined_dates['date_start'].min(), '—', combined_dates['date_end'].max())
print('Wymiar macierzy profilu:', combined_profile_matrix.shape)
print('Udział NaN w profilu [%]:', nan_percent(combined_profile_matrix))
if np.all(np.isnan(combined_profile_matrix)):
    raise ValueError('Wszystkie wartości profilu są NaN.')
print('\nPrędkość w profilu [m/year]:')
print('Minimum:', round(np.nanmin(combined_profile_matrix), 2))
print('Maksimum:', round(np.nanmax(combined_profile_matrix), 2))
print('Średnia:', round(np.nanmean(combined_profile_matrix), 2))
print('Mediana:', round(np.nanmedian(combined_profile_matrix), 2))
for name, values in combined_point_series.items():
    print(f'\nPunkt {point_labels[name]}:')
    print('Udział NaN [%]:', nan_percent(values))
    if not np.all(np.isnan(values)):
        print('Minimum [m/year]:', round(np.nanmin(values), 2))
        print('Maksimum [m/year]:', round(np.nanmax(values), 2))
distance_columns = [f'{distance:.1f}' for distance in distances]
profile_df = pd.DataFrame(combined_profile_matrix, columns=distance_columns)
profile_df.insert(0, 'source', combined_dates['source'])
profile_df.insert(0, 'date', combined_dates['date'])
profile_df.insert(0, 'date_end', combined_dates['date_end'])
profile_df.insert(0, 'date_start', combined_dates['date_start'])
profile_csv = output_dir / 'csv' / f'{glacier_name}_profile_velocity_2018-2025.csv'
profile_df.to_csv(profile_csv, index=False)
points_df = pd.DataFrame({'date_start': combined_dates['date_start'], 'date_end': combined_dates['date_end'], 'date': combined_dates['date'], 'source': combined_dates['source'], 'velocity_lower_m_y': combined_point_series['lower'], 'velocity_middle_m_y': combined_point_series['middle'], 'velocity_upper_m_y': combined_point_series['upper']})
points_csv = output_dir / 'csv' / f'{glacier_name}_points_equal_sections_velocity_2018-2025.csv'
points_df.to_csv(points_csv, index=False)
individual_point_csvs = {}
for name in point_paths:
    point_df = pd.DataFrame({'date_start': combined_dates['date_start'], 'date_end': combined_dates['date_end'], 'date': combined_dates['date'], 'source': combined_dates['source'], 'velocity_m_y': combined_point_series[name]})
    point_csv = output_dir / 'csv' / f'{glacier_name}_point_{name}_equal_section_velocity_2018-2025.csv'
    point_df.to_csv(point_csv, index=False)
    individual_point_csvs[name] = point_csv
print('\nZapisano CSV:')
print(profile_csv)
print(points_csv)
print(point_metadata_csv)
for name, path in individual_point_csvs.items():
    print(f'{name}: {path}')
vmin = 0
vmax = np.nanmax(combined_profile_matrix)
if not np.isfinite(vmax) or vmax <= 0:
    vmax = 1
velocity_cmap = plt.get_cmap('inferno_r').copy()
velocity_cmap.set_bad('white')
outline_xmin, outline_ymin, outline_xmax, outline_ymax = outline_gdf.total_bounds
outline_width = outline_xmax - outline_xmin
outline_height = outline_ymax - outline_ymin
if outline_width == 0:
    outline_width = 1000
if outline_height == 0:
    outline_height = 1000
margin_x = max(500, outline_width * 0.06)
margin_y = max(500, outline_height * 0.06)
all_gdf = gpd.GeoDataFrame(geometry=[outline_geom, profile_geom, *point_geoms.values()], crs=outline_gdf.crs)
all_xmin, all_ymin, all_xmax, all_ymax = all_gdf.total_bounds
xmin = min(outline_xmin - margin_x, all_xmin - 250)
xmax = max(outline_xmax + margin_x, all_xmax + 250)
ymin = min(outline_ymin - margin_y, all_ymin - 250)
ymax = max(outline_ymax + margin_y, all_ymax + 250)
map_xlim = (xmin, xmax)
map_ylim = (ymin, ymax)
fig = plt.figure(figsize=(15, 16))
gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[2.55, 2.55, 0.15, 1.35], hspace=0.38, left=0.09, right=0.94, top=0.965, bottom=0.06)
ax_map = fig.add_subplot(gs[0])
ax_heat = fig.add_subplot(gs[1])
cax = fig.add_subplot(gs[2])
ax_ts = fig.add_subplot(gs[3])
if basemap_img.ndim == 3:
    ax_map.imshow(basemap_img, extent=[basemap_bounds.left, basemap_bounds.right, basemap_bounds.bottom, basemap_bounds.top], origin='upper', zorder=1)
else:
    ax_map.imshow(basemap_img, extent=[basemap_bounds.left, basemap_bounds.right, basemap_bounds.bottom, basemap_bounds.top], origin='upper', cmap='gray', zorder=1)
outline_gdf.boundary.plot(ax=ax_map, color='blue', linewidth=1.0, zorder=4)
gpd.GeoSeries([profile_geom], crs=profile_gdf.crs).plot(ax=ax_map, color='red', linewidth=2.2, zorder=5)
for i, name in enumerate(['lower', 'middle', 'upper'], start=1):
    geom = point_geoms[name]
    ax_map.scatter(geom.x, geom.y, s=70, marker='o', facecolor='yellow', edgecolor='black', linewidth=1.0, zorder=6)
    ax_map.annotate(f'P{i}', (geom.x, geom.y), xytext=(7, 7), textcoords='offset points', fontsize=9, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black', alpha=0.8, pad=1.2), zorder=8)
ax_map.scatter(profile_start.x, profile_start.y, s=55, marker='o', facecolor='white', edgecolor='black', linewidth=0.9, zorder=7)
ax_map.scatter(profile_end.x, profile_end.y, s=55, marker='o', facecolor='white', edgecolor='black', linewidth=0.9, zorder=7)
dx = (map_xlim[1] - map_xlim[0]) * 0.012
dy = (map_ylim[1] - map_ylim[0]) * 0.012
ax_map.text(profile_start.x + dx, profile_start.y + dy, 'A', fontsize=11, fontweight='bold', color='black', zorder=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.18', alpha=0.85))
ax_map.text(profile_end.x + dx, profile_end.y - dy, 'B', fontsize=11, fontweight='bold', color='black', zorder=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.18', alpha=0.85))
ax_map.set_xlim(*map_xlim)
ax_map.set_ylim(*map_ylim)
ax_map.set_aspect('equal', adjustable='box')
ax_map.xaxis.tick_top()
ax_map.xaxis.set_label_position('top')
ax_map.tick_params(axis='y', right=True, labelright=True)
ax_map.tick_params(axis='both', labelsize=8)
ax_map.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:,.0f}'))
ax_map.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: f'{y:,.0f}'))
ax_map.set_xlabel('')
ax_map.set_ylabel('')
ax_map.set_title(f'{glacier_name} — overview map', fontsize=10, pad=10)
ax_map.text(0.99, 0.04, 'EPSG: 3413', transform=ax_map.transAxes, ha='right', va='bottom', fontsize=8, style='italic', bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.5))
im = None
for i in range(len(combined_dates)):
    row_values = combined_profile_matrix[i, :]
    row_masked = np.ma.masked_invalid(row_values)
    y_start = mdates.date2num(combined_dates.iloc[i]['date_start'])
    y_end = mdates.date2num(combined_dates.iloc[i]['date_end'])
    if y_end <= y_start:
        y_end = y_start + 1.0
    z = row_masked.reshape(1, -1)
    if len(distances) > 1:
        x_edges = np.empty(len(distances) + 1, dtype=float)
        x_edges[1:-1] = (distances[:-1] + distances[1:]) / 2
        x_edges[0] = distances[0] - (distances[1] - distances[0]) / 2
        x_edges[-1] = distances[-1] + (distances[-1] - distances[-2]) / 2
    else:
        x_edges = np.array([distances[0] - spacing / 2, distances[0] + spacing / 2], dtype=float)
    y_edges = np.array([y_start, y_end], dtype=float)
    current_im = ax_heat.pcolormesh(x_edges, y_edges, z, shading='flat', cmap=velocity_cmap, vmin=vmin, vmax=vmax, rasterized=True)
    if im is None:
        im = current_im
ax_heat.set_xlim(distances.min(), distances.max())
ax_heat.set_ylim(mdates.date2num(combined_dates['date_start'].min()), mdates.date2num(combined_dates['date_end'].max()))
ax_heat.set_ylabel('Time', fontsize=11)
ax_heat.set_xlabel('Distance along profile [m]', fontsize=11, labelpad=7)
ax_heat.yaxis_date()
ax_heat.yaxis.set_major_locator(mdates.YearLocator())
ax_heat.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax_heat.yaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 5, 9]))
ax_heat.yaxis.set_minor_formatter(mdates.DateFormatter('%m'))
ax_heat.tick_params(axis='y', which='major', labelsize=8)
ax_heat.tick_params(axis='y', which='minor', labelsize=7, pad=2)
ax_heat.xaxis.set_major_locator(plt.MultipleLocator(1000))
ax_heat.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:,.0f}'))
ax_heat.tick_params(axis='x', labelsize=8)
for spine in ax_heat.spines.values():
    spine.set_linewidth(1.0)
ax_heat.text(distances.min(), 1.02, 'A', transform=ax_heat.get_xaxis_transform(), ha='left', va='bottom', fontsize=11, fontweight='bold')
ax_heat.text(distances.max(), 1.02, 'B', transform=ax_heat.get_xaxis_transform(), ha='right', va='bottom', fontsize=11, fontweight='bold')
for boundary in [profile_length / 3.0, 2.0 * profile_length / 3.0]:
    ax_heat.axvline(boundary, linewidth=0.9, linestyle=':', alpha=0.8)
for i, name in enumerate(['lower', 'middle', 'upper'], start=1):
    x = profile_geom.project(point_geoms[name])
    ax_heat.axvline(x, linewidth=0.8, linestyle='--', alpha=0.65)
    ax_heat.text(x, 0.01, f'P{i}', transform=ax_heat.get_xaxis_transform(), ha='center', va='bottom', fontsize=8, fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=0.8))
cb = plt.colorbar(im, cax=cax, orientation='horizontal')
cb.set_label('Velocity [m/year]', fontsize=9, labelpad=5)
cb.set_ticks([0, vmax / 4, vmax / 2, 3 * vmax / 4, vmax])
cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:,.0f}'))
cb.ax.tick_params(labelsize=8, pad=2)
for name in ['lower', 'middle', 'upper']:
    values = combined_point_series[name]
    valid = np.isfinite(values)
    ax_ts.scatter(combined_dates.loc[valid, 'date'], values[valid], s=18, label=point_labels[name], zorder=3)
ax_ts.set_xlim(combined_dates['date'].min(), combined_dates['date'].max())
all_point_values = np.concatenate(list(combined_point_series.values()))
valid_all = np.isfinite(all_point_values)
if np.any(valid_all):
    point_vmax = np.nanmax(all_point_values)
    if point_vmax > 0:
        ax_ts.set_ylim(0, point_vmax * 1.08)
ax_ts.set_ylabel('Velocity [m/year]', fontsize=10)
ax_ts.set_xlabel('Time', fontsize=10)
ax_ts.set_title(f'{glacier_name} velocity in three equal-profile-section points', fontsize=10, pad=12)
ax_ts.xaxis.set_major_locator(mdates.YearLocator())
ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax_ts.grid(alpha=0.25)
ax_ts.legend(fontsize=8)
figure_png = output_dir / 'figures' / f'{glacier_name}_velocity_3_equal_sections_2018-2025.png'
figure_pdf = output_dir / 'figures' / f'{glacier_name}_velocity_3_equal_sections_2018-2025.pdf'
plt.savefig(figure_png, dpi=300, bbox_inches='tight')
plt.savefig(figure_pdf, bbox_inches='tight')
print('\n====================================')
print('ZAPISANO')
print('====================================')
print('Figura PNG:', figure_png)
print('Figura PDF:', figure_pdf)
print('CSV profil:', profile_csv)
print('CSV 3 punkty:', points_csv)
print('CSV metadata punktów w równych częściach profilu:', point_metadata_csv)
for name, path in individual_point_csvs.items():
    print(f'CSV {name}: {path}')
plt.show()
