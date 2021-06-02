#First we'll import all the module we need
from marvin.tools.maps import Maps

import pathlib
import pandas as pd
import numpy as np
from numpy.linalg import multi_dot

import sys
sys.path.insert(0, '/home/sshamsi_haverford_edu/galaxy_zoo/GZ3D_production') #this might need changing if working across platforms

import gz3d_fits

# The Spiral Galaxy class is defined below
class SpiralGalaxy(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.filename = self.file_path.split('/')[-1]
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
        
        
    def __repr__(self):
        return 'MaNGA ID {}'.format(self.mangaid)
            
    
    def check_usability(self, threshold = 5, pix_percentage = 1.5):
        image_spiral_mask = self.data.spiral_mask
        pixels_above_threshold = (image_spiral_mask >= threshold).sum()
                
        if (pixels_above_threshold * 100 / image_spiral_mask.size < pix_percentage):
            return False
        
        return True
    
    
    def make_emmasks(self):
        '''Takes masks from the MaNGA maps and if a spaxel is flagges as
        "DO NOT USE (2**30) then it marks it as "0". Creates global maek objects
        from these.'''
        
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
        '''Goes through all spaxels and creates a global array of spaxel distances from the map centre'''
        
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
    
        
    def update_spirals(self, spiral_threshold=3, other_threshold=3, ret_spiral_bool=False):
        '''Do you want to change the parametres for what does/doesn't count as a spaxel or a
        non-spaxel? Use this function to simply update the "Spiral" column of the global DF'''
        
        self.data.make_all_spaxel_masks(grid_size = self.map_shape)
        
        center_mask_spaxel_bool = self.data.center_mask_spaxel > other_threshold
        star_mask_spaxel_bool = self.data.star_mask_spaxel > other_threshold
        bar_mask_spaxel_bool = self.data.bar_mask_spaxel > other_threshold
        spiral_mask_spaxel_bool = self.data.spiral_mask_spaxel > spiral_threshold
        
        combined_mask = center_mask_spaxel_bool | star_mask_spaxel_bool | bar_mask_spaxel_bool
        
        spiral_spaxel_bool = spiral_mask_spaxel_bool & (~combined_mask)
        
        if ret_spiral_bool:
            return spiral_spaxel_bool
        
        self.spiral_spaxel_bool = spiral_spaxel_bool
        self.df['Spiral Arm'] = spiral_spaxel_bool.flatten()
        
    
    def load_btp_masks(self):
        '''Simple make arrays of indicating the BPT classification of each spxel.
        Then we append this to the global DF later.'''
        
        if not self.bpt_masks_loaded:
            bpt_masks = self.maps.get_bpt(return_figure=False, show_plot=False)

            self.comp = bpt_masks['comp']['global']
            self.agn = bpt_masks['agn']['global']
            self.seyfert = bpt_masks['seyfert']['global']
            self.liner = bpt_masks['liner']['global']
            
            self.bpt_masks_loaded = True
    
    
    def flux2sfr(self, ha_flux, ha_stdv, hb_flux, hb_stdv, avg=False):
        '''Take an H-alpha and H-beta flux, and then make an SFR measurement out of them.'''
        
        ha_flux = ha_flux * 1E-13
        hb_flux = hb_flux * 1E-13
        
        ha_stdv = ha_stdv * 1E-13
        hb_stdv = hb_stdv * 1E-13
        
        sfr = (self.delta * (ha_flux**3.36) * (hb_flux**-2.36))
        sfr_stdv = np.sqrt((3.36 * self.delta * (ha_flux**2.36) * (hb_flux**-2.36) * ha_stdv)**2 +
                           (-2.36 * self.delta * (ha_flux**3.36) * (hb_flux**-3.36) * hb_stdv)**2)
        
        if avg:
            return sfr / self.spax_area, sfr_stdv / self.spax_area
        
        return sfr, sfr_stdv
    
    
    def form_global_df(self, spiral_threshold=3, other_threshold=3):
        '''Make a global DF.'''
        
        if not self.global_df_loaded:
            self.load_btp_masks()
            self.make_r_array()
            self.make_emmasks()
            
            ha_array = self.hamap.value.flatten()
            sig_ha_array = self.hamap.error.value.flatten()
            ha_snr = self.hamap.snr.flatten()
            
            hb_array = self.hbmap.value.flatten()
            sig_hb_array = self.hbmap.error.value.flatten()
            hb_snr = self.hbmap.snr.flatten()
            
            comp_array = self.comp.flatten()
            agn_array = self.agn.flatten()
            seyfert_array = self.seyfert.flatten()
            liner_array = self.liner.flatten()
            
            data_array = np.array([self.r_array, ha_array, sig_ha_array, ha_snr, hb_array, sig_hb_array, hb_snr,
                                   comp_array, agn_array, seyfert_array, liner_array]).transpose()
            
            df = pd.DataFrame(data=data_array, columns=['Radius', '$H_{\\alpha}$', '$\sigma H_{\\alpha}$',
                                                        'S/N $H_{\\alpha}$', '$H_{\\beta}$', '$\sigma H_{\\beta}$',
                                                        'S/N $H_{\\beta}$', 'Comp', 'AGN', 'Seyfert', 'Liner'])
            
            df['$r/r_e$'] = df['Radius'] / self.eff_rad
            
            df.iloc[self.ha_mask_array, df.columns.get_loc('$H_{\\alpha}$')] = np.nan
            df.iloc[self.ha_mask_array, df.columns.get_loc('$\sigma H_{\\alpha}$')] = np.nan
            df.iloc[self.ha_mask_array, df.columns.get_loc('$H_{\\beta}$')] = np.nan
            df.iloc[self.ha_mask_array, df.columns.get_loc('$\sigma H_{\\beta}$')] = np.nan
            
            df.iloc[self.hb_mask_array, df.columns.get_loc('$H_{\\beta}$')] = np.nan
            df.iloc[self.hb_mask_array, df.columns.get_loc('$\sigma H_{\\beta}$')] = np.nan
            
            df = df.replace([np.inf, -np.inf], np.nan)
            
            self.df = df
            self.update_spirals(spiral_threshold=spiral_threshold, other_threshold=other_threshold)
            
            self.global_df_loaded = True
            
            
    def cov_matrix_maker(self, err_series):
        '''Calculates the covaraince matrix for our galaxy.That is rather intensive.'''
        
        corr_matrix = np.load('/raid5/homes/sshamsi/galaxy_zoo/GZ3D_spiral_analysis/Matrices/corr_matrices/corr_matrix' + str(self.map_shape[0]) + '.npy')
        
        r = self.hamap.size
        cov_mat = np.zeros((r, r))
        
        for item, frame in err_series.iteritems():
            if pd.isnull(frame):
                k = 0
            else:
                k = frame
                
            cov_mat[item] = corr_matrix[item] * k
            cov_mat[:, item] = corr_matrix[:, item] * k
        
        return cov_mat
    
    
    def make_cov_matrices(self, mode=None):
        '''This method loads and returns the H-a/H-b covariance matrix. If not available,
        it calculates it, which is resource intensive.'''
        
        if mode == None:
            raise ValueError('Argument "mode" must be set to "ha" or "hb".')
            
        elif mode == 'ha':
            hafile = pathlib.Path("Matrices/cov_matrices/" + self.filename.split('.')[0] + 'ha.npy')
            print(hafile)
            
            if hafile.exists():
                ha_cov = np.load(hafile)
                return ha_cov
            
            else:
                print ("H-a covariance file does not exist. Calculating...")
                ha_cov = self.cov_matrix_maker(self.df['$\sigma H_{\\alpha}$'])
                return ha_cov
        
        elif mode == 'hb':
            hbfile = pathlib.Path("Matrices/cov_matrices/" + self.filename.split('.')[0] + 'hb.npy')
            print(hbfile)
            
            if hbfile.exists():
                hb_cov = np.load(hbfile)
                return hb_cov
            
            else:
                print ("H-b covariance file does not exist. Calculating...")
                hb_cov = self.cov_matrix_maker(self.df['$\sigma H_{\\beta}$'])
                return hb_cov
            
                            
    def get_sfr(self, index, avg=False):
        '''Return the SFR for a bin of spxels.'''
        
        ha_flux, ha_stdv = self.get_emission(index, mode='ha', avg=avg)
        hb_flux, hb_stdv = self.get_emission(index, mode='hb', avg=avg)
        
        return self.flux2sfr(ha_flux, ha_stdv, hb_flux, hb_stdv, avg=avg)
    
    
    def get_emission(self, index, mode=None, avg=False):
        '''Return the H-a or H-b flux.'''
        
        self.form_global_df()
        
        set_index = set(index)
        tot_index = list(self.df.index)
        
        w_vec = np.array([[x in set_index for x in tot_index]]) * 1
        
        if mode == None:
            raise ValueError('Argument "mode" must be "ha", or "hb".')
        elif mode == 'ha':
            summ = self.df.loc[index.tolist(), '$H_{\\alpha}$'].sum()
            
            ha_cov = self.make_cov_matrices(mode=mode)
            cov_mat = ha_cov
        elif mode == 'hb':
            summ = self.df.loc[index.tolist(), '$H_{\\beta}$'].sum()
            
            hb_cov = self.make_cov_matrices(mode=mode)
            cov_mat = hb_cov
            
        var = np.linalg.multi_dot([w_vec, cov_mat, w_vec.T])[0][0]
        
        if avg:
            n = len(index)
            return summ / n, np.sqrt(var / (n**2))
        
        return summ, np.sqrt(var)