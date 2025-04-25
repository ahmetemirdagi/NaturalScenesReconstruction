import sys
import numpy as np
import sklearn.linear_model as skl
import pickle
import argparse

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub", help="Subject Number", default=1)
args = parser.parse_args()
sub = int(args.sub)
assert sub == 1  # We're only working with subject 1 for now

# Load fMRI data
train_path = 'data/processed_data/subj{:02d}/nsd_train_fmriavg_nsdgeneral_sub{}.npy'.format(sub, sub)
train_fmri = np.load(train_path)
test_path = 'data/processed_data/subj{:02d}/nsd_test_fmriavg_nsdgeneral_sub{}.npy'.format(sub, sub)
test_fmri = np.load(test_path)

# Preprocessing fMRI (following the same preprocessing as other implementations)
train_fmri = train_fmri/300
test_fmri = test_fmri/300

# Normalize fMRI data
norm_mean_train = np.mean(train_fmri, axis=0)
norm_scale_train = np.std(train_fmri, axis=0, ddof=1)
train_fmri = (train_fmri - norm_mean_train) / norm_scale_train
test_fmri = (test_fmri - norm_mean_train) / norm_scale_train

print("fMRI data statistics:")
print("Train - Mean:", np.mean(train_fmri), "Std:", np.std(train_fmri))
print("Test - Mean:", np.mean(test_fmri), "Std:", np.std(test_fmri))
print("Train - Max:", np.max(train_fmri), "Min:", np.min(train_fmri))
print("Test - Max:", np.max(test_fmri), "Min:", np.min(test_fmri))

num_voxels = train_fmri.shape[1]

# Load SD features
train_sd = np.load('data/extracted_features/subj{:02d}/nsd_sd_train.npy'.format(sub))
test_sd = np.load('data/extracted_features/subj{:02d}/nsd_sd_test.npy'.format(sub))

# Print shapes for debugging
print("\nShape information:")
print("Train fMRI shape:", train_fmri.shape)
print("Test fMRI shape:", test_fmri.shape)
print("Train SD features shape:", train_sd.shape)
print("Test SD features shape:", test_sd.shape)

# Reshape features if needed
if len(train_sd.shape) > 2:
    num_samples = train_sd.shape[0]
    feature_dim = np.prod(train_sd.shape[1:])
    train_sd = train_sd.reshape(num_samples, feature_dim)
    test_sd = test_sd.reshape(test_sd.shape[0], feature_dim)
else:
    num_samples, feature_dim = train_sd.shape

print("\nAfter reshaping:")
print("Train SD features shape:", train_sd.shape)
print("Test SD features shape:", test_sd.shape)

print("\nTraining Regression")
print("Number of voxels:", num_voxels)
print("Feature dimension:", feature_dim)

# Train regression model
reg = skl.Ridge(alpha=50000, max_iter=10000, fit_intercept=True)
reg.fit(train_fmri, train_sd)

# Make predictions
pred_test_features = reg.predict(test_fmri)

# Normalize predictions
std_norm_test_features = (pred_test_features - np.mean(pred_test_features, axis=0)) / np.std(pred_test_features, axis=0)
pred_features = std_norm_test_features * np.std(train_sd, axis=0) + np.mean(train_sd, axis=0)

# Calculate and print R² score
score = reg.score(test_fmri, test_sd)
print("\nRegression R² score:", score)

# Reshape predictions back to original shape if needed
if len(test_sd.shape) > 2:
    pred_features = pred_features.reshape(test_sd.shape)

# Save predictions
np.save('data/predicted_features/subj{:02d}/nsd_sd_predtest_nsdgeneral.npy'.format(sub), pred_features)

# Save regression weights
datadict = {
    'weight': reg.coef_,
    'bias': reg.intercept_,
}

with open('data/regression_weights/subj{:02d}/sd_regression_weights.pkl'.format(sub), "wb") as f:
    pickle.dump(datadict, f)

print("\nSaved predictions and regression weights")