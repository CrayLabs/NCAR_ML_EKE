from os import listdir, path
from netCDF4 import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler

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
            
        files = [file for file in listdir(datapath) if '.nc' in file]
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
        
        