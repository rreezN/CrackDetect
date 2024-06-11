# To Do

## Model, Training and Feature Extraction
- [ ] Cross validation
  - Idea for solution: KFold when creating features (using Platoon), store the data under e.g. MultiRocketMV_50000 > Fold1 > statistics / features / targets
- [ ] Smaller hidden layer (30 neurons)
- [ ] Add additional hidden layer
- [ ] MSE confine to 0:15 (did not understand this)
- [ ] Don't tanh, normalize it or use raw output instead
- [ ] Shapley values (?)
- [ ] When training the model, validation losses are insane...
  - Seems to happen because some features from multirocket go to extreme values (e+32), primarily channels: 4515, 4526, 4531, 4842
  - [x] Try: Normalize instead of standardize
    - Did not fix it :(
  - [x] Try univariate multirocket
    - Did not fix it :()
  - [ ] Come up with something else to fix it
  - [ ] Investigate multirocket initial kernels of exploding channels
    - [ ] Plot Input signals over time along with features over time of the channels
  - [ ] Normalize channels so they sum to 1 (to avoid exploding channels)
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