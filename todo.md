# To Do

## Model, Training and Feature Extraction
- [ ] Create Feature Data Loader
- [ ] Implement new data loader
- [x] Feature extraction script
  - [x] Save features to file
    - [x] MultiRocket Features
    - [x] Hydra Features
    - [x] Save Max and Min targets
    - [x] Calculate and save statistics on features (mean, std)
      - [x] Hydra Statistics
      - [x] MultiRocket Statistics
- [ ] Targets need to be in same range ([0, 1]?)
  - [ ] Load Max and Min
  - [ ] ```new_target = ((target - Min) / (Max - Min))```

 
- [ ] Train script needs to batch over features
- [ ] Train script needs to run epochs over data
- [ ] Train script needs to follow Hydra/MR implementation
  
## Predicting
- [ ] Modify predict script to follow train script
- [ ] Save features to file
- [ ] Load features for prediction (avoid long predict inference time)
