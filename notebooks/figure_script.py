#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#init_notebook_mode(connected=True)

from scipy.stats import norm

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance, DMatrix

from tqdm import tqdm
from pickle import dump, load


# In[2]:


import ml_eke
#import tensorflow as tf
#from tensorflow import keras


# # Data loading
# 
# We need to load the samples. `columns` is a list in which we keep all needed predictands.

# In[3]:


columns1 =  [
            'MKE_z',
            'MKE_sfc',
            'slope_z',
            'relative_vorticity_z',
            'divergence_z',
            'deformation_z',
            'relative_vorticity_sfc',
            'divergence_sfc',
            'deformation_sfc',
            'Rd_dx_z'] 

columns2 = ['grad_SSH_sfc']

columns = columns1+columns2

def get_samples(start, end, columns, model_data):

    samples = []
    targets = []
    for sample in tqdm(range(start, end), desc="Extracting samples"):
        X, Y, mask = model_data.extract_sample_from_time(predictors=columns, predictands=['EKE_sfc'], sample_idx=sample)
        samples.append(pd.DataFrame(X.data, columns=columns))
        targets.append(pd.DataFrame(Y.data, columns=["EKE"]))

    data = pd.concat(samples).reset_index()
    data.drop("index", inplace=True, axis=1)
    targets = pd.concat(targets).reset_index()
    targets.drop("index", inplace=True, axis=1)
    data["EKE"] = targets["EKE"]
    
    return data

def get_samples_2D(start, end, columns, model_data):
    
    # get first sample and infer size
    samples, targets, masks = model_data.extract_sample_from_time_2D(predictors=columns,predictands = ['EKE_sfc'], sample_idx = start)
    samples = np.expand_dims(samples, axis=0)
    targets = np.expand_dims(targets, axis=0)
    masks   = np.expand_dims(masks, axis=0)
    
    for sample in tqdm(range(start+1, end), desc="Extracting 2D samples"):
        X, Y, mask = model_data.extract_sample_from_time_2D(predictors=columns,predictands = ['EKE_sfc'], sample_idx = sample)
        samples = np.append(samples, [X], axis=0)
        targets = np.append(targets, [Y], axis=0)
        masks   = np.append(masks,   [mask], axis=0)
        
    return samples, targets, masks


# # Pre-processing
# 
# We perform a two-step preproces for the predictors.
# 
# ## First pre-processing step: logarithm where needed
# - The feature `Rd_dx_z` does not need any pre-processing. We keep it as last feature, and our `regularize` function relies on this convention (i.e. the last feature is not processed).
# - Some features are log-normally distributed. They are always positive, so we can simply take their logarithm, to obtain normally-distributed features. We apply this technique to the predictand feature (`EKE`) too.
# - Other features follow what looks like a Laplace distribution with some outliers. In this case, we apply $f(x) = (\log(\left|x\right|)+36)*\text{sign}(x)$. This functions splits the negative and positive domains, compresses the range, is monotonic, and bijective, if and only if $|x|>10^{-15}$. Here is a plot of the function over $\mathbb{R}$.
# 

# In[4]:


# f = lambda x: np.log(np.abs(x)+36.0)*np.sign(x)
# x_plot = np.arange(-100, 100, 1e-2)

# plt.plot(x_plot, f(x_plot), '.')
# plt.draw()


# In[5]:


def regularize(x):
    
    y = x.copy()
    
    if y.ndim > 1:
        for i in tqdm(range(y.shape[-1]), desc="Preprocessing features"): 
            channel = y[:,i]
            notnan = channel[~np.isnan(channel)]
            if np.all(notnan>0):
                y[:,i] = np.log(y[:,i])
            else:
                zeros = y[:,i]==0
                signs = np.sign(y[:,i])
                y[zeros] = 1 # avoid nan errors
                y[:,i] = np.log(np.abs(y[:,i]))*signs+signs*36.0
                y[zeros] = 0
    else:
        notnan = y[~np.isnan(y)]
        if np.all(notnan>0):
            y = np.log(y)
        else:
            zeros = y==0
            y = np.log(np.abs(y))*np.sign(y)+np.sign(y)*36.0
            y[zeros] = 0
    return y


# ## Second pre-processing step: Scaling
# 
# The second pre-processing step is the same for all predictors, we scale the features so that they have approximately zero-mean and standard deviation of 1: this is usually good for Neural Network training. We do not apply this step to the predictand. We will store the scaler object, as it will be need to pre-process data at inference time.

# In[6]:


def prep_samples(dataset, scaler=None):

    print("Dropping empty values...")
    samples = dataset.dropna()
    
    print("Dropping negative EKE samples...")
    # Negative EKE values seem to be more of an artifact than anything else
    samples = samples[samples['EKE']>0]
    
    targets = samples["EKE"]
    print("Removing EKE from features...")
    samples.drop("EKE", inplace=True, axis=1)
    
    X = regularize(samples.values)
    
    Y = np.log(targets.values)
    
    # scale the data
    if scaler is None:
        print("Fitting scaler and scaling data...")
        scaler = RobustScaler(quantile_range=(25.0, 75.0))
        X = scaler.fit_transform(X)
    else:
        print("Scaling data with provided scaler...")
        X = scaler.transform(X)
    
    X = pd.DataFrame(X, columns=columns)
    
    return X, Y, scaler    


# Once the data is pre-processed, we split it into train and validation (called test here) sets.

# In[7]:


def prep_data(dataset, scaler=None):
    data, targets, scaler = prep_samples(dataset, scaler)
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)
    print('Dimensions of the training feature matrix: {}'.format(X_train.shape))
    print('Dimensions of the training target vector: {}'.format(y_train.shape))
    print('Dimensions of the test feature matrix: {}'.format(X_test.shape))
    print('Dimensions of the test target vector: {}'.format(y_test.shape))
    return X_train, X_test, y_train, y_test, scaler
    


# There are two different datasets, one (`cf10`) has values averaged over 10x10 boxes, the other one (`cf2`) is averaged over 2x2 boxes. For each type, we have 120 samples. We only use 110 samples for each type, keeping 10 as test values, which we will use as tests for the trained models.

# In[19]:

# UNCOMMENT THIS [19,20] IF YOU NEED TO CREATE A NEW VERSION OF THE DATASET!

# datapath1 = '/lus/scratch/spartee/MOM6_1-10/'

# datapath2 = '/lus/scratch/spartee/MOM6_1-10_data_with_ssh/'

# model_data = ml_eke.pop_data(datapath1, datapath1, skip_vars = ['x','y','depth','depth_stdev'], extra_pref='cf2')
# model_data.extend_inventory(datapath2)
# samples = get_samples(0, 5, columns, model_data)
# for cf_idx in range(3, 11):
#     model_data = ml_eke.pop_data(datapath1, datapath1, skip_vars = ['x','y','depth','depth_stdev'], extra_pref=f'cf{cf_idx}')
#     model_data.extend_inventory(datapath2)
#     print(f'cf{cf_idx}')
#     scale_factor = cf_idx/2
#     samples_loc = get_samples(0, min(int(np.ceil(5*scale_factor*scale_factor)), 110), columns, model_data)
#     samples = pd.concat([samples,samples_loc])


# In[20]:


# X_train, X_test, y_train, y_test, scaler = prep_data(samples)


# ## Visual inspection of pre-processing results
# 
# Let's have a look at the resulting feature distributions. 

# In[27]:

# UNCOMMENT THIS [27,28] FOR ANALYSIS OF DATASET

# plt.figure(figsize=(10,20))
# for i in range(X_train.shape[-1]):
#     sample = X_train.values[:, i]
#     min_bin = np.min(np.min(sample))
#     max_bin = np.max(np.max(sample))
#     print('---------------------------------')
#     print(f'min {columns[i]} = {np.min(sample)}, max {columns[i]} = {np.max(sample)}')
#     print('---------------------------------')
#     plt.subplot(len(columns)//2+len(columns)%2,2,i+1)
#     plt.hist(sample, bins=1000, density=True)
#     plt.title(columns[i])
#     plt.draw()
    
# print('---------------------------------')
# print(f'min EKE = {np.min(y_train)}, max EKE = {np.max(y_train)}')
# print('---------------------------------')
# plt.figure(figsize=(10,4))
# plt.hist(y_train, bins=1000, density=True, alpha=0.5)
# plt.hist(y_test, bins=1000, density=True, alpha=0.5)
# plt.legend(['train', 'validation'])
# plt.title('EKE')
# plt.draw()


# We notice that the predictand is approximately log-normally distributed. This means that we have way less samples for the values belonging to the two tales, and when we will train the neural network against mean squared error, the extremum values will therefore have less weight. We explored a possible way to mitigate this phenomenon, that is inverse sampling weighting. We approximated the distribution with a Gaussian, and we multiplied the loss function terms by the inverse of the probability density for the predictand values. This custom loss can be found in the `loss.py` file. In the plot below, we see how well the Gaussian distribution fits the train (and validation) sets.

# In[28]:


# plt.figure(figsize=(10,4))

# plt.hist(y_train, bins=1000, density=True, alpha=0.5)
# xmin, xmax = plt.xlim()
# x_pdf = np.linspace(xmin, xmax, 100)
# mu, std = norm.fit(y_train)
# p = norm.pdf(x_pdf, mu, std)
# plt.plot(x_pdf, p, 'k', linewidth=2)

# plt.legend([f'Norm dist $\mu={mu:.4f}$, $\sigma={std:.4f}$'])
# plt.title('EKE')
# plt.draw()


# In[29]:

# SET TO TRUE TO SAVE...
save = False

if save:
    dump(scaler, open('ml_eke/nn/data/scaler_all.pkl', 'wb'))

    np.save('ml_eke/nn/data/y_train_all', y_train)
    np.save('ml_eke/nn/data/y_test_all', y_test)

    np.save('ml_eke/nn/data/X_train_all', X_train.values)
    np.save('ml_eke/nn/data/X_test_all', X_test.values)


# We will now train an XGBoost regressor, to be able to compare the results of our Neural Network(s) to it. The dataset is too large for the algorithm to run in a reasonable time, thus we will use 1/100th of the data points. 

# In[50]:


# If you have the data, just skip the loading and pre-processing
load_data = False
if load_data:
    X_train = pd.DataFrame(np.load('./ml_eke/nn/data/X_train_sfc.npy'), columns=columns)
    X_test =  pd.DataFrame(np.load('./ml_eke/nn/data/X_test_sfc.npy'), columns=columns)

    y_train = np.load('./ml_eke/nn/data/y_train_sfc.npy')
    y_test = np.load('./ml_eke/nn/data/y_test_sfc.npy')

# UNCOMMENT THIS (AND [51]) IF YOU NEED XGBOOST BASELINE
    
# rf = XGBRegressor()
# num_samples = X_train.shape[0]
# rf.fit(X_train[:num_samples//100], y_train[:num_samples//100]) 
# rf.save_model('./ml_eke/nn/data/xgbreg_cf10')
# y_train_rf = rf.predict(X_train[:num_samples//100])
# y_test_rf = rf.predict(X_test)
# rf_results = pd.DataFrame({'algorithm':['XGBRegressor'],
#             'training error': [mean_squared_error(y_train[:num_samples//100], y_train_rf)],
#             'test error': [mean_squared_error(y_test, y_test_rf)],
#             'training_r2_score': [r2_score(y_train[:num_samples//100], y_train_rf)],
#             'test_r2_score': [r2_score(y_test, y_test_rf)]})
# rf_results


# The regressor can give us an important insight into the data, we can see what the most relevant features are.

# In[51]:


# plot_importance(rf)

# plt.draw()


# We can see that when a feature is available both as surface value (`_sfc`) or averaged along the vertical axis (`_z`), the two features seem to have the same importance. We tried using only one of the two quantities, but the error was slightly larger, thus we defer this to future studies.

# # Inference and analysis
# 
# In this section, we load two trained neural networks and compare the results with the XGBoost solution. We load the models with `compile=False`, because we do not need to train them. The Neural Network `model_cus` uses the custom loss, whereas `model_mse` uses a standard mean squared error loss. We also load a scaler we previously saved: **notice that this scaler must be the same one used to scale the data we trained our NNs on, otherwise results will be inconsistent**.

# In[8]:


import torch
#from nn_models import EKEResnet
from torchsummary import summary

model_mse = torch.load('./ml_eke/nn/pytorch/ResNet_11-30_mse_all.pkl', map_location=torch.device('cpu'))
model_cus = torch.load('./ml_eke/nn/pytorch/ResNet_11-100_custom_all.pkl', map_location=torch.device('cpu'))
#print(model_mse)
summary(model_mse, (11,))

#model_mse = keras.models.load_model('ml_eke/nn/trained_models/GEN-INT_l2_0.002_8_2020-12-17_512_mse.h5', compile=False)
#model_cus = keras.models.load_model('ml_eke/nn/trained_models/GEN-INT_l2_0.002_8_2020-12-17_512_custom.h5', compile=False)
#model_mse.summary()


# There seems to be a problem with the scaler. Best way is to generate a new scaler (i.e. do not use a pre-saved one)
# when you prepare the samples

# scaler = load(open('ml_eke/nn/data/scaler_all.pkl', 'rb'))
# rf = XGBRegressor()
# rf.load_model('./ml_eke/nn/data/xgbreg_cf10')


# We load the datasets we did not use to train our NNs and look at how the predicted value distributions compare to the target one.

# In[25]:

# Change these to the two paths with columns1 and columns2 features in the files. If you do not need two dirs,
# just take out extend_inventory
# In this case, we do inference on the 1/10 test set

datapath1 = '/lus/scratch/spartee/MOM6_1-10/'
datapath2 = '/lus/scratch/spartee/MOM6_1-10_data_with_ssh/'


model_data = ml_eke.pop_data(datapath1, datapath1, skip_vars = ['x','y','depth','depth_stdev'], extra_pref='cf8')
model_data.extend_inventory(datapath2)

dataset_10 = get_samples(110, 111, columns, model_data)
#dataset_2 = get_samples(110, 120, columns, model_data, 'cf2')
data_10, targets_10, _ = prep_samples(dataset_10)
#data_2, targets_2, _ = prep_samples(dataset_2, scaler)
data_10


# In[36]:


# make some predictions
preds_cus = model_cus(torch.tensor(data_10.values))
preds_cus = preds_cus.detach().numpy()
preds_mse = model_mse(torch.tensor(data_10.values))
preds_mse = preds_mse.detach().numpy()
#preds_rf  = rf.predict(data_10).reshape((-1,1))


# show true distribution vs predicted
plt.figure(figsize=(10,6))
plt.hist(targets_10,bins=100,alpha=0.3,label='True distribution', density=True)
plt.hist(preds_cus,bins=100,alpha=0.3,label='Predicted Custom', density=True)
plt.hist(preds_mse,bins=100,alpha=0.3,label='Predicted MSE', density=True)
#plt.hist(preds_rf,bins=100,alpha=0.3,label='Predicted XGB', density=True, color='green')
plt.legend()
plt.draw()


# In[37]:


# plt.figure(figsize=(10,20))
# for i in range(data_10.shape[-1]):
#     sample = data_10.values[:, i]
#     min_bin = np.min(np.min(sample))
#     max_bin = np.max(np.max(sample))
#     print('---------------------------------')
#     print(f'min {columns[i]} = {np.min(sample)}, max {columns[i]} = {np.max(sample)}')
#     print('---------------------------------')
#     plt.subplot(len(columns)//2+len(columns)%2,2,i+1)
#     plt.hist(sample, bins=1000, density=True, alpha=0.5)
#     sample = X_train.values[:, i]
#     plt.hist(sample, bins=1000, density=True, alpha=0.5)
#     plt.title(columns[i])
#     plt.draw()
    
# print('---------------------------------')
# print(f'min EKE = {np.min(targets_10)}, max EKE = {np.max(targets_10)}')
# print('---------------------------------')
# plt.figure(figsize=(10,4))
# plt.hist(targets_10, bins=1000, density=True, alpha=0.5)
# plt.legend(['train', 'validation'])
# plt.title('EKE')
# plt.draw()


# With the newest data, it seems like XGB and the MSE model are pretty similar in outcome, whereas the custom loss performs better on the tails of the distribution, but poorly on the peak.
# 
# To display the results in a meaningful way, we keep the original samples as maps, instead of collecting single point values. 

# In[38]:


def prep_maps(sample, target, mask, scaler=None, model=None):
    sample_shape = sample.shape
    target_shape = target.shape
    
    X = sample.reshape(-1,sample_shape[-1])
    Y = target.reshape(-1,1)
    mask_flat = mask.reshape(-1,1)
    
    samples = pd.DataFrame(X, columns=columns)
    targets = pd.DataFrame(Y, columns=["EKE"])

    samples["EKE"] = targets["EKE"]
    
    data_out, targets_out, _ = prep_samples(samples)#, scaler)
    
    if isinstance(model, XGBRegressor):
        pred = model.predict(data_out)
    else:
        pred = model(torch.tensor(data_out.values))
        pred = pred.detach().numpy()
        
    
    data_map = np.zeros(sample_shape)
    for channel in range(sample_shape[-1]):
        map_channel = data_map[:,:,:,channel:channel+1]
        map_channel[mask] = data_out.values[:,channel:channel+1]
        map_channel[~mask] = np.nan
        
    targets_map = np.zeros(target_shape)
    targets_map[mask,0] = targets_out
    targets_map[~mask,0] = np.nan
    
    pred_map = np.zeros(target_shape)
    pred_map[mask,0] = pred.squeeze()
    pred_map[~mask,0] = np.nan
    
    
    return data_map, targets_map.squeeze(), pred_map.squeeze()
    


# In[44]:

# Change these to the two paths with columns1 and columns2 features in the files. If you do not need two dirs,
# just take out extend_inventory
# In this case, we do inference on the 1/10 test set

datapath1 = '/lus/scratch/spartee/MOM6_1-10/'
datapath2 = '/lus/scratch/spartee/MOM6_1-10_data_with_ssh/'


model_data = ml_eke.pop_data(datapath1, datapath1, skip_vars = ['x','y','depth','depth_stdev'], extra_pref='cf10')
model_data.extend_inventory(datapath2)


sample_10, target_10, mask_10 = get_samples_2D(110, 111, columns, model_data)
_, _, disp_pred_10_cus = prep_maps(sample_10, target_10, mask_10, model=model_cus)
disp_sample_10, disp_target_10, disp_pred_10_mse = prep_maps(sample_10, target_10, mask_10, model=model_mse)
#disp_sample_10, disp_target_10,  disp_pred_10_rf = prep_maps(sample_10, target_10, mask_10, scaler, rf)

# print('2x2')
# sample_2,  target_2,  mask_2  = get_samples_2D(110, 111, 'cf2')
# disp_sample_2, disp_target_2, disp_pred_2_cus = prep_maps(sample_2, target_2, mask_2, scaler, model_cus)
# _, _,                         disp_pred_2_mse = prep_maps(sample_2, target_2, mask_2, scaler, model_mse)
# _, _,                         disp_pred_2_rf = prep_maps(sample_2, target_2, mask_2, scaler, rf)


# We start by plotting the predicted values for the two different NNs. We can see that they perform better on the high-resolution dataset, less on the coarser one. Two possible reasons are: the ratio between the number of fine- and coarse-grained samples is ~25:1, this means that the network will be trained mostly on high-resolution samples; the information (and thus the prediction) might be more accurate when less averaging takes place.

# In[46]:


#mask_2_s  = mask_2.squeeze()
mask_10_s = mask_10.squeeze()

plt.figure(figsize=(16,12))

vmin = min(np.min(disp_target_10[mask_10_s]),
        np.min(disp_pred_10_mse[mask_10_s]),
        np.min(disp_pred_10_cus[mask_10_s]))
vmax = max(np.max(disp_target_10[mask_10_s]),
        np.max(disp_pred_10_mse[mask_10_s]),
        np.max(disp_pred_10_cus[mask_10_s]))

plt.subplot(2,3,1)
plt.pcolormesh(disp_target_10, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.title('Target 4x4')
plt.subplot(2,3,2)
plt.pcolormesh(disp_pred_10_mse, vmin=vmin, vmax=vmax)
plt.title('MSE Prediction')
plt.colorbar()
plt.subplot(2,3,3)
plt.pcolormesh(disp_pred_10_cus, vmin=vmin, vmax=vmax)
plt.title('Custom loss Prediction')

# vmin = min(np.min(disp_target_2[mask_2_s]),
#            np.min(disp_pred_2_mse[mask_2_s]),
#            np.min(disp_pred_2_cus[mask_2_s]))
# vmax = max(np.max(disp_target_2[mask_2_s]),
#            np.max(disp_pred_2_mse[mask_2_s]),
#            np.max(disp_pred_2_cus[mask_2_s]))
# plt.subplot(2,3,4)
# plt.pcolormesh(disp_target_2, vmin=vmin, vmax=vmax)
# plt.title('Target 2x2')
# plt.subplot(2,3,5)
# plt.pcolormesh(disp_pred_2_mse, vmin=vmin, vmax=vmax)
# plt.title('MSE Prediction')
# plt.subplot(2,3,6)
# plt.pcolormesh(disp_pred_2_cus, vmin=vmin, vmax=vmax)
# plt.title('Custom loss Prediction')
# plt.draw()

plt.savefig('predicted_vals_maps.png')


# We plot the error for the three models (MSE, custom loss, and XGBoost). We see again that MSE and XGBoost are almost indistinguishable. 

# In[14]:


plt.figure(figsize=(16,12))

disp_err_10_mse = disp_target_10 - disp_pred_10_mse
disp_err_10_cus = disp_target_10 - disp_pred_10_cus
# disp_err_10_rf = disp_target_10 - disp_pred_10_rf

#disp_err_2_mse = disp_target_2 - disp_pred_2_mse
#disp_err_2_cus = disp_target_2 - disp_pred_2_cus
# disp_err_2_rf = disp_target_2 - disp_pred_2_rf

vmin = min(np.min(disp_err_10_mse[mask_10_s]),
           np.min(disp_err_10_cus[mask_10_s]))
vmax = max(np.max(disp_err_10_mse[mask_10_s]),
           np.max(disp_err_10_cus[mask_10_s]))
plt.subplot(1,2,1)
plt.pcolormesh(disp_err_10_mse, vmin=vmin, vmax=vmax)
plt.title('MSE Error')
plt.subplot(1,2,2)
plt.pcolormesh(disp_err_10_cus, vmin=vmin, vmax=vmax)
plt.title('Custom loss Error')


# In[16]:


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.hist(disp_err_10_mse[mask_10_s].reshape((-1,1)), bins=100, density=True, label='MSE', alpha=0.3)
plt.hist(disp_err_10_cus[mask_10_s].reshape((-1,1)), bins=100, density=True, label='custom', alpha=0.3)
#plt.hist(disp_err_10_rf[mask_10_s].reshape((-1,1)), bins=100, density=True, label='XGB', alpha=0.3)
plt.title('10x10 averaging')
plt.legend()

# plt.subplot(1,2,2)
# plt.hist(disp_err_2_mse[mask_2_s].reshape((-1,1)), bins=100, density=True, label='MSE', alpha=0.3)
# plt.hist(disp_err_2_cus[mask_2_s].reshape((-1,1)), bins=100, density=True, label='custom', alpha=0.3)
# plt.hist(disp_err_2_rf[mask_2_s].reshape((-1,1)), bins=100, density=True, label='XGB', alpha=0.3)
# plt.title('2x2 averaging')
# plt.legend()
# plt.draw()

plt.savefig('error_hist.png')

# We see that in terms of error, MSE achieves the best distribution, a narrower peak around 0 and, as expected, the 2x2 averaged data is predicted more accurately.
# 
# As a historical remark, we note that the CNN was adapted from the one we used with a previous version of the data, which did not have the `Rd_dx_z` feature and predictions were less accurate. With this new feature, it is possible that a slimmer model would be as efficient as the CNN, and require less computation power.
