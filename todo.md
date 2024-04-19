# To Do

## Model, Training and Feature Extraction
- [X] Create Feature Data Set
  - [x] Implement transforms
    - [x] Normalize each feature
      - [x] Rocket feature transformation
      - [X] Problem: Some stds are 0 (current solution: pick max(stds, 1e-12), but causes features and loss to explode)
      - [x] New solution: Mean over feature_means and feature_stds instead
    - [x] Targets need to be in same range ([0, 1]?)
      - [x] Load Max and Min
      - [x] ```new_target = ((target - Min) / (Max - Min))```
      - [ ] Check that everything works
- [X] Implement new data loader
- [x] Feature extraction script
  - [x] Save features to file
    - [x] MultiRocket Features
    - [x] Hydra Features
    - [x] Save Max and Min targets
    - [x] Calculate and save statistics on features (mean, std)
      - [x] Hydra Statistics
      - [x] MultiRocket Statistics

 
- [x] Train script needs to batch over features
- [x] Train script needs to run epochs over data
- [x] Saving best model during training
- [x] Train script needs to follow Hydra/MR implementation
- [ ] Figure out which signals to extract features from
  
## Predicting
- [x] Modify predict script to follow train script
- [x] Load features for prediction (avoid long predict inference time)
- [ ] Figure out what is happening with correlations...
