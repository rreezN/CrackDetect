# crackdetect

To pull files using dvc:
```
dvc pull
```

# TODO:

- Cut all data to fit the trip

- Overview of sampling rates for each attribute of all trips/measurements

- Create linear interpolation of all attributes of all trips/measurements as a function of timestamp
- Plot interpolated functions for different trips/cars

### For friday (16/02)

1. All data must be interpolated by distance driven

2. For Autopi GPS this implies calculating distance from time and speed. <br> 
Note that gps, accl and gyro do not sample at the same frequency. Hence, speed must be interpolated by time to align gps, accl and gyro. <br>
Longitude and latitude are gound in Autopi Gopro GPS as column 3 and 4 respectively. Time is column 2 and 2D and 3D speed is column 6 and 7. Time is always column 2 for Autopi Gopro data.

3. Investigate how Aran data must be aligned. Are there a distance measure or do we also need to calculate that.

4. Compare the aligned data from P79/Aran to each trip of the Green Mobility cars. (Drifts???) 

5. profit???


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
