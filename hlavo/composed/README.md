# Interaction between 1D and 3D model

## 1D model

## 3D model
Input(from 1D queue): location_id, longitude, latitude, date_time [np.datetime64[m]], velocity [m/s]

### 1D input dict initialization:
Inputs:
- longitude, latitude for corners of the computational rectangle
- (longitude, latitude) for all 1D models

Output:
dictionary, key = location_id:
value: dataclass Projection_1Dmodel
    - longitude
    - latitude
    - base_function_filed .. np.array shape=(nx, ny); computed using Voronoi -> with one/zero values
    

