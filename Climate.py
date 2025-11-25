import numpy as np
import matplotlib.pyplot as plt
import requests
import zipfile
import io
import geopandas as gpd
import s3fs
import zarr
import dask 
import dask.array
import xarray as xr
import pandas as pd
from shapely.geometry import Point
from shapely.prepared import prep
import rioxarray
from shapely import points


first_year = 2020
last_year = 2023
downsample_factor = 63


# Task 1 : Download the Texas shape file.

url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_5m.zip"
response = requests.get(url)

if response.status_code == 200:
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall("path_to_extract")  # م

# بارگذاری فایل‌های استخراج شده
shapefile_path = "path_to_extract/cb_2023_us_state_5m.shp"
all_states = gpd.read_file(shapefile_path)
texas_shape = all_states[all_states['NAME'] == 'Texas'].to_crs('EPSG:4326')



base_url = 'noaa-nws-aorc-v1-1-1km' 
dataset_years = list(range(first_year, last_year))
s3_out = s3fs.S3FileSystem(anon=True)

store = [s3fs.S3Map(
            root=f"{base_url}/{dataset_year}.zarr", 
            s3=s3_out,
            check=False
        ) for dataset_year in dataset_years]

#############################################################

base_url = 'noaa-nws-aorc-v1-1-1km'
s3_out = s3fs.S3FileSystem(anon=True)

texas_bbox = texas_shape.bounds
minx, miny, maxx, maxy = texas_bbox.iloc[0]

print(f"Texas bounding box: {minx}, {miny}, {maxx}, {maxy}")

available_years = []
for year in range(first_year, last_year):
    if s3_out.exists(f"{base_url}/{year}.zarr"):
        available_years.append(year)

print(f"Available years: {available_years}")
texas_datasets = []
for year in available_years:
    print(f"\nProcessing year: {year}")
    
    try:
        store = s3fs.S3Map(root=f"{base_url}/{year}.zarr", s3=s3_out, check=False)
        ds = xr.open_dataset(store, engine='zarr', chunks={'time': 720}) 
        
        ds_texas_bbox = ds.sel(
            longitude=slice(minx - 1, maxx + 1),
            latitude=slice(miny - 1, maxy + 1)
        )
        
        ds_texas_bbox = ds_texas_bbox.rio.write_crs("EPSG:4326")
        ds_texas = ds_texas_bbox.rio.clip(texas_shape.geometry, drop=True)
        texas_datasets.append(ds_texas)  
    except Exception as e:
        print(f"Error processing year {year}: {e}")


ds_texas = xr.concat(texas_datasets, dim='time')
#####################################################################
## Task 2: Label grids by numbering them from 1 to ~700,000. 
# You should start from top left of Texas to bottom right.


with dask.config.set(array_slicing={"split_large_chunks": False}):
    stacked_ds = ds_texas.stack(grid=("latitude", "longitude"))

num_grids = stacked_ds.grid.size
print(f"Total number of grids (before masking): {num_grids}")


###################################################################
nrows, ncols = len(ds_texas.latitude), len(ds_texas.longitude)
grid_labels = np.arange(1, num_grids + 1)
labels_2d = grid_labels.reshape((nrows, ncols))

# Fix latitude orientation
lat_ascending = (ds_texas.latitude.values[1] > ds_texas.latitude.values[0])
if lat_ascending:
    labels_2d = np.flipud(labels_2d)

label_da = xr.DataArray(
    labels_2d,
    coords={"latitude": ds_texas.latitude, "longitude": ds_texas.longitude},
    dims=["latitude", "longitude"]
)

# Use union_all instead of unary_union
texas_geom = texas_shape.union_all()
texas_geom_prep = prep(texas_geom)

# Vectorized shapely evaluation (MUCH faster)
lon, lat = np.meshgrid(ds_texas.longitude.values, ds_texas.latitude.values)
pts = points(lon.ravel(), lat.ravel())                 # Vectorized point creation
mask_flat = np.array([texas_geom_prep.contains(p) for p in pts])
mask = mask_flat.reshape(lon.shape)


masked_labels = label_da.where(mask)
num_valid_grids = int(masked_labels.notnull().sum().values)
print(f"Number of grid cells inside Texas (after masking): {num_valid_grids}")


################################################################
### Verification that the grids are labelled as innstructed

first_label_val = masked_labels.min().item()
last_label_val = masked_labels.max().item()

first_label_cell = masked_labels.where(masked_labels == first_label_val, drop=True)
lat_1, lon_1 = first_label_cell.latitude.values[0], first_label_cell.longitude.values[0]

last_label_cell = masked_labels.where(masked_labels == last_label_val, drop=True)
lat_last, lon_last = last_label_cell.latitude.values[0], last_label_cell.longitude.values[0]


### Task 3 : Change the spatial resolution from 800m to 50km. This requires keeping grid # 1, 
# counting to grid # 51 and removing grids between 1 and 51. 
# Repeat the same procedure all the way towards the last grid, which is grid # ~700,000. 
# After that should have nearly 280 grids.


coarsened_labels = masked_labels.isel(
    latitude=slice(0, None, downsample_factor),
    longitude=slice(0, None, downsample_factor)
)

num_selected = int(coarsened_labels.notnull().sum().values)
print(f"Number of selected grids for Task 3: {num_selected}")

selected_lons, selected_lats = np.meshgrid(
    coarsened_labels.longitude.values,
    coarsened_labels.latitude.values
)

valid_mask = ~np.isnan(coarsened_labels.values)


# 1. ساخت شبکه مختصات
selected_lons, selected_lats = np.meshgrid(
    coarsened_labels.longitude.values,
    coarsened_labels.latitude.values)

# 2. بررسی valid_mask
valid_mask = ~np.isnan(coarsened_labels.values)
print(f"تعداد مقادیر معتبر: {np.sum(valid_mask)}")

# 3. فیلتر کردن داده‌های معتبر
valid_lons = selected_lons[valid_mask]
valid_lats = selected_lats[valid_mask]

# 4. چاپ داده‌ها
print(valid_lons[:10])  # اولین ۱۰ مقدار از longitude‌های معتبر
print(valid_lats[:10])  # اولین ۱۰ مقدار از latitude‌های معتبر

### Task 4 : 
# Precipitation data is hourly based. At all ~280 grids, 
# sum precipitation data in each day to switch to daily-based data.

labeled_ds = ds_texas.where(mask)
labeled_ds['grid_label'] = label_da.where(mask)
downsample_factor = 63

ds_50km = labeled_ds.isel(
    latitude=slice(0, None, downsample_factor),
    longitude=slice(0, None, downsample_factor)
)

ds_50km = ds_50km.where(ds_50km['grid_label'].notnull(), drop=True)

ds_50km['APCP_surface']


###########################################



selected_lons_flat = selected_lons.flatten()
selected_lats_flat = selected_lats.flatten()

output_file = "hourly_temperature_data.csv"
batch_size = 10 

with open(output_file, 'w') as f:
    f.write("longitude,latitude,datetime,temperature\n")

print(f"Processing {len(selected_lons_flat)} points in batches of {batch_size}...")

total_records = 0
for batch_start in range(0, len(selected_lons_flat), batch_size):
    batch_end = min(batch_start + batch_size, len(selected_lons_flat))
    batch_points = selected_lons_flat[batch_start:batch_end]
    batch_lats = selected_lats_flat[batch_start:batch_end]
    
    print(f"Processing batch: points {batch_start} to {batch_end-1}")
    
    batch_records = 0
    for j, (lon, lat) in enumerate(zip(batch_points, batch_lats)):
        point_idx = batch_start + j
        
        try:
            point_data = ds_texas.sel(longitude=lon, latitude=lat, method='nearest')
            hourly_temps = point_data['TMP_2maboveground']
            temp_series = hourly_temps.to_series()
            with open(output_file, 'a') as f:
                for datetime, temp in temp_series.items():
                    datetime_str = datetime.strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"{lon:.4f},{lat:.4f},{datetime_str},{temp:.2f}\n")
                    batch_records += 1
            
            print(f"  Point {point_idx}: {len(temp_series)} records")
            del point_data, hourly_temps, temp_series
            
        except Exception as e:
            print(f"  Error at point {point_idx}: {e}")
    
    total_records += batch_records
    print(f"Batch complete: {batch_records} records (Total: {total_records})")

print(f"\nProcessing complete! {total_records} total records saved to {output_file}")

