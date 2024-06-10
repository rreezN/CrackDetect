# To Do

## Model, Training and Feature Extraction
- [ ] When training the model, validation losses are insane...
  - Seems to happen because some features from multirocket go to extreme values (e+32), primarily channels: 4515, 4526, 4531, 4842
  - [x] Try: Normalize instead of standardize
    - Did not fix it :(
  - [x] Try univariate multirocket
    - Did not fix it :()
  - [ ] Come up with something else to fix it
- [X] Optimize feature mean, avoid storing all features in memory
  - [X] Consider further investigating accuracy of Welford's Online Algorithm for calculating means and stds 
  - [ ] Consider doing similar optimisations to the kpi statistics (doesn't seem to be necessary ATM)
- [ ] Figure out which signals to extract features from
- [ ] Max_num_channels in HydraMV (????)
- [ ] Extract new features, using more than acc only
  - [x] Check if this fixes std = 0 issue
  - It does not
- [X] Normalize inputs before feature extraction
- [ ] Find out why some features have std = 0
  - [x] Also find out if for different signals they always output the same, i.e. channel 13 is always 14.7 or 
  - They do not
  - [ ] If can't find out and fix, then remove features with std = 0 before training
- [ ] Reduce step-size to avoid loss explosion (gradients exploding?)
  - [ ] If this doesn't work then we can do gradient clipping
- [ ] Some form of logging (wandb, hydra, whatever)

## Experiments
- [ ] Test out various signals for feature extraction
  - [ ] All accelerations
  - [ ] Only z-acceleration
  - [ ] All signals that Asmus pointed out (all signals in matlab file)
  - [ ] Self-selected features
- [ ] Test out various models
  - [ ] Deep network
  - [ ] Shallow network
  - [ ] Large hidden layers
  - [ ] MSE vs MAE
  - [ ] Other ML tricks :)
- [ ] Model using only Hydra or only MultiRocket features
- [ ] Hyperparameter tuning (learning rate)
- [ ] LR Schedulers
- [ ] Dropout?

## Aflevering
- [ ] Comments, comments, comments... and more comments
- [ ] Type hints
- [ ] NO MAGIC NUMBERS (TOMMY BLIVER KED AF DET)
- [ ] Henvis til papers i koden (ligninger, osv.)
- [ ] Notebooks
- [ ] README: tilføj afsnit om check_hdf5.py