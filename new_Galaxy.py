from marvin.tools.maps import Maps

import pandas as pd
import numpy as np
from numpy.linalg import multi_dot

import sys
sys.path.insert(0, '/raid5/homes/sshamsi/galaxy_zoo/GZ3D_production')

import gz3d_fits

class SpiralGalaxy(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = gz3d_fits.gz3d_fits(file_path)
        self.mangaid = self.data.metadata['MANGAID'][0]
        
        self.maps = Maps(self.mangaid)
        self.hamap = self.maps.emline_gflux_ha_6564
        self.hbmap = self.maps.emline_gflux_hb_4862
        self.eff_rad = self.maps.nsa['elpetro_th50_r'] * 2
        self.redshift = self.maps.nsa['z']
        self.mass = self.maps.nsa['sersic_mass']
        self.lgmass = np.log10(self.mass)
        self.elpetro_ba = self.maps.nsa['elpetro_ba']
        self.theta = np.radians(self.maps.nsa['elpetro_phi'] - 90.0)
        self.map_shape = self.hamap.shape
        
        self.d_mpc = ((299792.458 * self.redshift) / 70) #Mpc
        self.d_kpc = self.d_mpc * 1E3
        self.d_m = self.d_mpc * 3.085677581E+22 # m
        self.delta = (4 * np.pi * (self.d_m**2)) / ((2.8**2.36) * (10**41.1))
        self.spax_area = (0.0000024240684055477 * self.d_kpc)**2
        
        self.global_df_loaded = False
        self.bpt_masks_loaded = False
        self.r_array_loaded = False
        self.coor_matrix_loaded = False
        self.sfr_cov_loaded = False
        self.ha_cov_loaded = False
        
        
    def __repr__(self):
        return 'MaNGA ID {}'.format(self.mangaid)
    
    
    def form_corellation_matrix(self):
        if not self.coor_matrix_loaded:
            shape_list = np.load('shape_list.npy')

            for i in range(len(shape_list)):
                if shape_list[i][0] == self.map_shape[0]:
                    index = i

            self.corr_matrix = np.load('coor_matrix_list.npy')[index]
            
            self.coor_matrix_loaded = True
    
    
    def check_usability(self, threshold = 5, pix_percentage = 1.5):
        image_spiral_mask = self.data.spiral_mask
        pixels_above_threshold = (image_spiral_mask >= threshold).sum()
                
        if (pixels_above_threshold * 100 / image_spiral_mask.size < pix_percentage):
            return False
        
        return True
    
    
    def make_emmasks(self):
        self.ha_mask_array = self.hamap.mask.flatten()
        self.hb_mask_array = self.hbmap.mask.flatten()
        
        for i in range(len(self.ha_mask_array)):
            if self.ha_mask_array[i] & 1073741824 == 0:
                self.ha_mask_array[i] = 0
                self.hb_mask_array[i] = 0
            else:
                self.ha_mask_array[i] = 1
                if self.hb_mask_array[i] & 1073741824 == 0:
                    self.hb_mask_array[i] = 0
                else:
                    self.hb_mask_array[i] = 1
                    
        self.ha_mask_array = np.array(self.ha_mask_array, dtype=bool)
        self.hb_mask_array = np.array(self.hb_mask_array, dtype=bool)
        
        
    def make_r_array(self):
        if not self.r_array_loaded:
            r_array = np.array([])

            a, b = self.map_shape
            k, h = (a - 1) / 2.0, (b - 1) / 2.0 #map centre

            for y, x in [(y, x) for y in range(a) for x in range(b)]:
                j, i = (-1 * (y - k), x - h) #vector from centre

                spax_angle = (np.arctan(j / i)) - self.theta
                vec_len = (j**2.0 + i**2.0)**0.5
                r = vec_len * ((np.cos(spax_angle))**2.0 + ((np.sin(spax_angle))/self.elpetro_ba)**2.0)**0.5

                r_array = np.append(r_array, r)

            self.r_array = r_array
            
            self.r_array_loaded = True
    
        
    def get_arms_spaxel_mask(self, mask_threshold=3):
        self.data.make_all_spaxel_masks(grid_size = self.map_shape)
        self.arms_spaxel_mask = self.data.spiral_mask_spaxel > mask_threshold
        
    
    def load_btp_masks(self):
        if not self.bpt_masks_loaded:
            bpt_masks = self.maps.get_bpt(return_figure=False, show_plot=False)

            self.comp = bpt_masks['comp']['global']
            self.agn = bpt_masks['agn']['global']
            self.seyfert = bpt_masks['seyfert']['global']
            self.liner = bpt_masks['liner']['global']
            
            self.bpt_masks_loaded = True
    
    
    def flux2sfr(self, x):
        ha_flux = x['$H_{\\alpha}$'] * 1E-13
        hb_flux = x['$H_{\\beta}$'] * 1E-13
        
        ha_stdv = x['$\sigma H_{\\alpha}$'] * 1E-13
        hb_stdv = x['$\sigma H_{\\beta}$'] * 1E-13
        
        x['SFR'] = (self.delta * (ha_flux**3.36) * (hb_flux**-2.36)) / self.spax_area
        x['$\sigma$SFR'] = np.sqrt((3.36 * self.delta * (ha_flux**2.36) * (hb_flux**-2.36) * ha_stdv)**2 +
                                   (-2.36 * self.delta * (ha_flux**3.36) * (hb_flux**-3.36) * hb_stdv)**2) / self.spax_area
        
        return x
    
    
    def form_global_df(self):
        if not self.global_df_loaded:
            self.get_arms_spaxel_mask()
            self.load_btp_masks()
            self.make_r_array()
            self.make_emmasks()

            ha_array = self.hamap.value.flatten()
            sig_ha_array = self.hamap.error.value.flatten()

            hb_array = self.hbmap.value.flatten()
            sig_hb_array = self.hbmap.error.value.flatten()

            spax_type_array = self.arms_spaxel_mask.flatten()

            comp_array = self.comp.flatten()
            agn_array = self.agn.flatten()
            seyfert_array = self.seyfert.flatten()
            liner_array = self.liner.flatten()

            data_array = np.array([self.r_array, ha_array, sig_ha_array, hb_array, sig_hb_array, spax_type_array,
                                   comp_array, agn_array, seyfert_array, liner_array]).transpose()

            df = pd.DataFrame(data=data_array, columns=['Radius', '$H_{\\alpha}$', '$\sigma H_{\\alpha}$', 
                                                        '$H_{\\beta}$', '$\sigma H_{\\beta}$', 'Spiral Arm',
                                                        'Comp', 'AGN', 'Seyfert', 'Liner'])

            df = df.replace([np.inf, -np.inf], np.nan)
            
            df.iloc[self.ha_mask_array, df.columns.get_loc('$H_{\\alpha}$')] = np.nan
            df.iloc[self.ha_mask_array, df.columns.get_loc('$\sigma H_{\\alpha}$')] = np.nan
            df.iloc[self.ha_mask_array, df.columns.get_loc('$H_{\\beta}$')] = np.nan
            df.iloc[self.ha_mask_array, df.columns.get_loc('$\sigma H_{\\beta}$')] = np.nan
            
            df.iloc[self.hb_mask_array, df.columns.get_loc('$H_{\\beta}$')] = np.nan
            df.iloc[self.hb_mask_array, df.columns.get_loc('$\sigma H_{\\beta}$')] = np.nan
            
            df = df.apply(self.flux2sfr, axis=1)
            
            df = df.replace([np.inf, -np.inf], np.nan)
            
            df['$r/r_e$'] = df['Radius'] / self.eff_rad
            
            self.df = df
            
            self.global_df_loaded = True
            
            
    def cov_matrix_maker(self, err_series):        
        r = self.hamap.size
        cov_mat = np.zeros((r, r))
        
        for item, frame in err_series.iteritems():
            if pd.isnull(frame):
                k = 0
            else:
                k = frame
                
            cov_mat[item] = self.corr_matrix[item] * k
            cov_mat[:, item] = self.corr_matrix[:, item] * k
        
        return cov_mat
    
    
    def make_cov_matrices(self, mode=None):
        self.form_corellation_matrix()
        
        if mode == None:
            raise ValueError('Argument "mode" must be set to "sfr" or "ha".')
            
        elif mode == 'sfr':
            if not self.sfr_cov_loaded:
                self.sfr_cov = self.cov_matrix_maker(self.df['$\sigma$SFR'])
                self.sfr_cov_loaded = True
        
        elif mode == 'ha':
            if not self.ha_cov_loaded:
                self.ha_cov = self.cov_matrix_maker(self.df['$\sigma H_{\\alpha}$'])
                self.ha_cov_loaded = True
                
                
    def get_var(self, index, mode=None, avg=False):
        index = set(index)
        tot_index = list(self.df.index)
        
        w_vec = np.array([[x in index for x in tot_index]]) * 1
                
        if mode == None:
            raise ValueError('Argument "mode" must be "sfr" or "ha"')
        elif mode == 'ha':
            self.make_cov_matrices(mode=mode)
            cov_mat = self.ha_cov
        elif mode == 'sfr':
            self.make_cov_matrices(mode=mode)
            cov_mat = self.sfr_cov
            
        var = np.linalg.multi_dot([w_vec, cov_mat, w_vec.T])[0][0]
        
        if avg:
            n = len(index)
            return var / (n**2)
        
        return var
    
    
    def integrate_sfr(self, mode='all', upper_limit=1, lower_limit=0.1, avg=False, sfr_lim=5, sig_sfr_lim=5, sfr_sig_cutoff=True):
        df = self.df
        
        df = df[(df['$r/r_e$'] <= upper_limit) & (df['$r/r_e$'] >= lower_limit)]
        df = df[(df.Comp == 0) & (df.AGN == 0) & (df.Seyfert == 0) & (df.Liner == 0)]
        
        if sfr_sig_cutoff:
            df = df.dropna()
            df = df[df['SFR'] >= df['$\sigma$SFR']]
        else:
            df.loc[df['SFR'] > sfr_lim, 'SFR'] = np.nan
            df.loc[df['$\sigma$SFR'] > sig_sfr_lim, '$\sigma$SFR'] = np.nan
            df = df.dropna()
                
        if mode == 'spirals':
            df = df[df['Spiral Arm'] == 1]
            
        if mode == 'non-spirals':
            df = df[df['Spiral Arm'] == 0]
        
        summ = df.SFR.sum()
        
        if avg:
            return summ / len(df.index), self.get_var(df.index, mode='sfr', avg=avg)
        
        return summ, self.get_var(df.index, mode='sfr')