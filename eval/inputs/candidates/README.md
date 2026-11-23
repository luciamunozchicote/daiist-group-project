# Candidate photos for eval/cases.csv

60 DeepSolarEye photos (Mehta et al., WACV 2018, CC BY-NC-SA) that the model has
NOT seen: not in the training data, not in eval/reference.csv, from the test days
(27-30 June 2017). 12 per band of true power loss.

The true loss is in the file name, after `L_` (0-1): `..._L_0.0994_I_0.37.jpg` = 9.9 % loss.
The cleaning threshold from the € formula is 9.3 % (economics.py):
above it the right action is `clean`, below it `wait`.

Pick the ones you want as cases, move them to eval/inputs/cases/ WITHOUT renaming
them (the app reads irradiance and hour from the name), and write their row in
eval/cases.csv. Delete this folder before the final tag.
