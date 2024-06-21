# To Do
- [ ] Report: Table der forklarer alle arguments til hvert script og hvordan de bliver brugt
- [ ] Flot figur over hdf5 fil struktur
- [ ] Gennemlæs draft paper 5.1 feature extraction
- [X] Main python fil der kører det hele
  - [X] Skal også validere noget i forhold til om output for den der lige har kørt er korrekt i forhold til det vi har af forventninger

## Model, Training and Feature Extraction
- [X] Post processering på predictions er ok (sørg for at alle predictions under 0 er 0)
- [X] Some form of logging (wandb, hydra, whatever)
- [ ] Consider what we want to do wrt. KPI scaling
  - Don't scale
  - Scale between 0, 1
  - Standardize (what we do now)
- [X] Max_num_channels in HydraMV (????)
- [ ] Don't tanh, normalize it or use raw output instead
  - [X] Don't tanh
  - [ ] Normalize it instead
- [X] Optimize feature mean, avoid storing all features in memory
  - [X] Consider further investigating accuracy of Welford's Online Algorithm for calculating means and stds 
  - [ ] Consider doing similar optimisations to the kpi statistics (doesn't seem to be necessary ATM)
- [ ] Find out why some features have std = 0
  - [x] Also find out if for different signals they always output the same, i.e. channel 13 is always 14.7 or 
  - They do not
  - [X] If can't find out and fix, then remove features with std = 0 before training
- [X] Smaller hidden layer (30 neurons)
- [X] Add additional hidden layer
- [X] Figure out which signals to extract features from
- [X] Extract new features, using more than acc only
  - [X] Check if this fixes std = 0 issue
  - It does not
- [X] Normalize inputs before feature extraction
- [X] When training the model, validation losses are insane...
- [X] Reduce step-size to avoid loss explosion (gradients exploding?)
- [X] Cross validation

## Experiments
- [ ] Test out various signals for feature extraction
  - [X] All accelerations
  - [X] Only z-acceleration
  - [X] All signals that Asmus pointed out (all signals in matlab file)
  - [X] Self-selected features
- [X] Test out various models
  - [X] Deep network
  - [X] Shallow network
  - [X] Large hidden layers
  - [X] MSE vs MAE
  - [X] Other ML tricks :)
- [X] Model using only Hydra or only MultiRocket features
- [X] Hyperparameter tuning (learning rate)
- [ ] LR Schedulers
- [X] Batch norm ?
- [X] Dropout?

## Aflevering
- [ ] ASSERTIONS i model delen
- [ ] Comments, comments, comments... and more comments
- [ ] Type hints
- [ ] NO MAGIC NUMBERS (TOMMY BLIVER KED AF DET)
- [ ] Henvis til papers i koden (ligninger, osv.)
- [ ] Notebooks
- [ ] README: tilføj afsnit om check_hdf5.py