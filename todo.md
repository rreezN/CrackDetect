# To Do

## Model and Training
- [ ] Implement new data loader
- [ ] Feature extraction script
  - [ ] Save features to file
    - [ ] MultiRocket Features
    - [ ] Hydra Features
    - [ ] Save `Max` and `Min` targets
    - [ ] Calculate and save statistics on features (`mean`, `std`)
      - [ ] Hydra Statistics
      - [ ] MultiRocket Statistics
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
