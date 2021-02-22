from os import listdir, path
from netCDF4 import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler

from scipy import ndimage as nd
import pandas as pd
from tqdm import tqdm

from xgboost import XGBRegressor, plot_importance, DMatrix

class pop_data:
    """Stores data and metadata of output from POP used to predict eddy kinetic energy"""
    
    def __init__(self,predictor_path, predictand_path, skip_vars = ['x','y','depth','depth_stdev'], extra_pref = None, first_suffix='_013_01.nc'):
        self.metafile = None
        self.extra_pref = extra_pref
        self.skip_vars = skip_vars
        self.first_suffix = first_suffix
        self.predictor_inventory = self._generate_inventory(predictor_path)
        self.predictand_inventory = self._generate_inventory(predictand_path)
        self.predictors = self.predictor_inventory.keys()
        self.predictands = self.predictand_inventory.keys()
        self.extract_meta_vars()
    
    
    def _generate_inventory(self, datapath):
        """
        Generate a mapping stored in a dictionary between available variables and where they can be found
        Inputs: 
            datapath: Path to directory containing outputs from a POP simulation
        """
    
        files = [file for file in listdir(datapath) if '.nc' in file and not 'xyz' in file]
        # file_prefixes = list(set([ file.split('_')[0] for file in files ]))
        # file_prefixes = list(set([ "_".join(file.split('_')[0:2]) for file in files ]))
        if self.extra_pref:
            file_prefixes = list(set([ "_".join(file.split('_')[0:2] + [self.extra_pref]) for file in files ]))
        else:
            file_prefixes = list(set([ "_".join(file.split('_')[0:2]) for file in files ]))
            
        inventory = {}
        for file_prefix in file_prefixes:
            fname = path.join(datapath,f'{file_prefix}{self.first_suffix}')
            if not self.metafile:
                self.metafile = fname
            vars = [ var for var in list(Dataset(fname).variables) if var not in self.skip_vars ]
            for var in vars:
                inventory[var] = {'files': sorted([path.join(datapath,file) 
                                           for file in listdir(datapath) if file_prefix in file])}
        return inventory

        
    def extend_inventory(self, datapath, variable_type='all', extra_pref=None, first_suffix=None):
        """
        Generate a mapping stored in a dictionary between available variables and where they can be found
        Inputs: 
            datapath: Path to directory containing outputs from a POP simulation
        """
        if extra_pref is None:
            extra_pref = self.extra_pref
        if first_suffix is None:
            first_suffix = self.first_suffix
            
        files = [file for file in listdir(datapath) if '.nc' in file and not 'xyz' in file]
        # file_prefixes = list(set([ file.split('_')[0] for file in files ]))
        # file_prefixes = list(set([ "_".join(file.split('_')[0:2]) for file in files ]))
        if extra_pref:
            file_prefixes = list(set([ "_".join(file.split('_')[0:2] + [extra_pref]) for file in files ]))
        else:
            file_prefixes = list(set([ "_".join(file.split('_')[0:2]) for file in files ]))
        
        inventory = {}
        for file_prefix in file_prefixes:
            fname = path.join(datapath,f'{file_prefix}{first_suffix}')
            if not self.metafile:
                self.metafile = fname
            vars = [ var for var in list(Dataset(fname).variables) if var not in self.skip_vars ]
            for var in vars:
                inventory[var] = {'files': sorted([path.join(datapath,file) 
                                           for file in listdir(datapath) if file_prefix in file])}
        
        if variable_type == 'predictors':
            self.predictor_inventory = {**self.predictor_inventory, **inventory}
            self.predictors = self.predictor_inventory.keys()
        elif variable_type == 'predictands':
            self.predictand_inventory = {**self.predictand_inventory, **inventory}
            self.predictands = self.predictand_inventory.keys()
        else:
            self.predictor_inventory = {**self.predictor_inventory, **inventory}
            self.predictors = self.predictor_inventory.keys()
            self.predictand_inventory = {**self.predictand_inventory, **inventory}
            self.predictands = self.predictand_inventory.keys()
    
    def extract_meta_vars(self):
        self.x = Dataset(self.metafile).variables['x'][:]
        self.y = Dataset(self.metafile).variables['y'][:]
        
    
    def extract_2d_var(self,varname,file_idx):
        """
        Load an individual 2D variable from a specific index
        """
        if varname in self.predictors:
            file = self.predictor_inventory[varname]['files'][file_idx]
        elif varname in self.predictands:
            file = self.predictand_inventory[varname]['files'][file_idx]
        else:
            raise ValueError(f'{varname} not a predictor or predictand')                
        
        return Dataset(file).variables[varname][:]
    
    
    def extract_sample_from_time(self, predictors=None, predictands=None, sample_idx=0):
        """
        Create a numpy array whose dimensions are consistent with the predictor arrays in Keras and scikit-learn
        """
        
        if not predictors:
            predictors = self.predictors
        if not predictands:
            predictands = self.predictors
        
        X = np.vstack([ self.extract_2d_var(var,sample_idx).reshape(-1) for var in predictors ])
        Y = np.vstack([ self.extract_2d_var(var,sample_idx).reshape(-1) for var in predictands ])
        
        is_valid = np.all(~np.isnan(X),axis=0) & np.all(~np.isnan(Y),axis=0)
        
        return X[:,is_valid].T, Y[:,is_valid].T, is_valid
        
        
    def extract_sample_from_time_2D(self, predictors=None, predictands=None, sample_idx=0):
        """
        Create a numpy tensor whose dimensions are consistent with the predictor arrays in Keras for CNNs
        """
        
        if not predictors:
            predictors = self.predictors
        if not predictands:
            predictands = self.predictors
        
        X = np.moveaxis(np.stack([ self.extract_2d_var(var,sample_idx) for var in predictors ]), 0, 2)
        Y = np.moveaxis(np.stack([ self.extract_2d_var(var,sample_idx) for var in predictands ]), 0, 2)
        
        is_valid = np.all(~np.isnan(X),axis=2) & np.all(~np.isnan(Y),axis=2)
        
        return X, Y, is_valid
        
        
    def extract_sample_from_time_dropzeros(self, predictors=None, predictands=None, sample_idx=0):
        """
        Create a numpy array whose dimensions are consistent with the predictor arrays in Keras and scikit-learn
        """
        
        if not predictors:
            predictors = self.predictors
        if not predictands:
            predictands = self.predictors
        
        X = np.vstack([ self.extract_2d_var(var,sample_idx).reshape(-1) for var in predictors ])
        Y = np.vstack([ self.extract_2d_var(var,sample_idx).reshape(-1) for var in predictands ])
        
        is_valid = np.all(~np.isnan(X),axis=0) & np.all(~np.isnan(Y),axis=0) & np.all(X>0, axis=0)
        
        return X[:,is_valid].T, Y[:,is_valid].T, is_valid    


# Notebook helpers:
    
def get_samples(start, end, columns, model_data, predictands=['EKE_sfc']):

    samples = []
    targets = []
    for sample in tqdm(range(start, end), desc="Extracting samples"):
        X, Y, mask = model_data.extract_sample_from_time(predictors=columns, predictands=predictands, sample_idx=sample)
        samples.append(pd.DataFrame(X.data, columns=columns))
        targets.append(pd.DataFrame(Y.data, columns=["EKE"]))

    data = pd.concat(samples).reset_index()
    data.drop("index", inplace=True, axis=1)
    targets = pd.concat(targets).reset_index()
    targets.drop("index", inplace=True, axis=1)
    data["EKE"] = targets["EKE"]
    
    return data


# Andrew Shao's wisdom
def helmholtz_smooth_transform(array, coeff=1, gauss_smooth=1):
    return nd.gaussian_filter(array + coeff*nd.laplace(array), gauss_smooth)

def regularize_2D(x, skip_col=None):
    
    y = x.copy()
    for i in range(y.shape[-1]):
        if skip_col is not None and skip_col[i]:
            continue 
        else:
            y[:,:,i] = helmholtz_smooth_transform(y[:,:,i], coeff=1)
    
    return y

def get_samples_2D(start, end, columns, model_data, predictands=['EKE_sfc'], preprocess=False, skip_preprocess_var=[]):
    
    # get first sample and infer size
    samples, targets, masks = model_data.extract_sample_from_time_2D(predictors=columns,predictands = predictands, sample_idx = start)
    skip_col = [columns[i] in skip_preprocess_var for i in range(len(columns))]
    
    samples = np.expand_dims(samples, axis=0)
    if preprocess:
        samples[0,:,:,:] = regularize_2D(samples[0,:,:,:], skip_col)
    targets = np.expand_dims(targets, axis=0)
    masks   = np.expand_dims(masks, axis=0)
    
    
    for sample in tqdm(range(start+1, end), desc="Extracting 2D samples"):
        X, Y, mask = model_data.extract_sample_from_time_2D(predictors=columns,predictands = predictands, sample_idx = sample)
        if preprocess:
            samples = np.append(samples, [regularize_2D(X, skip_col)], axis=0)
        else:
            samples = np.append(samples, [X], axis=0)
        targets = np.append(targets, [Y], axis=0)
        masks   = np.append(masks,   [mask], axis=0)
        
    return samples, targets, masks     

def maps_to_array(X, Y):
    X = np.reshape(X, [-1, X.shape[-1]])
    Y = np.reshape(Y, [-1,1])
    samples = pd.DataFrame(X, columns=columns)
    samples["EKE"] = Y
    
    return samples

def regularize(x, columns, skip_col=[], abs_val=None):
    
    y = x.copy()
      
    if y.ndim > 1:
        for i in range(y.shape[-1]): 
            if skip_col[i]:
                print(f'id({columns[i]})')
                continue
            if abs_val is not None:
                use_abs_val = abs_val[i]
            else:
                channel = y[:,i]
                notnan = channel[~np.isnan(channel)]
                if np.all(notnan>0):
                    use_abs_val = False
                else:
                    use_abs_val = True
                    
            if not use_abs_val:
                print(f'log({columns[i]})')
                y[:,i] = np.log(y[:,i])
            else:
                print(f'log(abs({columns[i]}))*sign')
                zeros = y[:,i]==0
                signs = np.sign(y[:,i])
                y[zeros,i] = 1  # just to avoid divide by zero
                y[:,i] = (np.log(np.abs(y[:,i]))+36.0)*signs
                y[zeros,i] = 0
    else:
        notnan = y[~np.isnan(y)]
        if np.all(notnan>0):
            y = np.log(y)
        else:
            zeros = y==0
            y[zeros] = 1  # just to avoid divide by zero
            y = (np.log(np.abs(y))+36)*np.sign(y)
            y[zeros] = 0
    return y

def prep_samples(dataset, columns, scaler=None, skip_vars=[], abs_val=None, clean_after_reg=False, scale=True):

    print("Dropping empty values...")
    samples = dataset.dropna()
    
    print("Dropping negative EKE samples...")
    # Negative EKE values seem to be more of an artifact than anything else
    samples = samples[samples['EKE']>0]
    
    targets = samples["EKE"].values.copy()
    print("Removing EKE from features...")
    samples.drop("EKE", inplace=True, axis=1)
    
    skip_col = [columns[i] in skip_vars for i in range(len(samples.columns))]
    
    X = regularize(samples.values, columns, skip_col, abs_val=abs_val)
    
    if clean_after_reg:
        X[~np.isfinite(X)] = 0
    
    Y = np.log(targets)
    
    
    if scale:
        # scale the data
        if scaler is None:
            print("Fitting scaler and scaling data...")
            avg = np.float32(np.mean(X, axis=0, dtype=np.float64))
            std = np.float32(np.std(X, axis=0, dtype=np.float64))
            X = (X-avg)/std
            scaler = [avg, std]
        else:
            print("Scaling data with provided scaler...")
            X = (X-scaler[0,:])/scaler[1,:]
    
    X = pd.DataFrame(X, columns=columns)
    
    return X, Y, scaler  

def prep_maps(sample, target, mask, columns, scaler, model, predict_fn, abs_val=None, clean_after_reg=True, skip_vars=None):
    sample_shape = sample.shape
    target_shape = target.shape
    
    X = sample.reshape(-1,sample_shape[-1])
    Y = target.reshape(-1,1)
    mask_flat = mask.reshape(-1,1)
    
    samples = pd.DataFrame(X, columns=columns)
    targets = pd.DataFrame(Y, columns=["EKE"])

    samples["EKE"] = targets["EKE"]
    
    data_out, targets_out, _ = prep_samples(samples, columns, scaler, abs_val=abs_val, clean_after_reg=clean_after_reg, skip_vars=skip_vars)
    
    if isinstance(model, XGBRegressor):
        pred = model.predict(data_out)
    else:
        pred = predict_fn(model, data_out)
    
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
   