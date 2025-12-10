"""
    @file:              DicomImage_class.py
    @Author:            Philippe Dionne et Louis-Bernard St-Cyr

    @Creation Date:     08/2025
    @Last modification: 09/2025

    @Description:       A modified version of the MTF code of Philippe Dionne, with the added options of computing the SNR and other metrics. Designed for single shot EPI DWI sequences.
"""


import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.fft import fft, fftfreq
from numpy.fft import fft2, ifft2, fftshift
from matplotlib.widgets import RectangleSelector, Button, EllipseSelector
import csv
import pickle as pkl
from pylinac.core.nps import noise_power_spectrum_1d
from scipy.interpolate import interp1d
import time
import datetime
from typing import Optional
import itertools
from ADC_MRI_extractor import ADC_MRI_extractor
import re
import itertools


class DicomImage:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = r'\usepackage[charter]{mathdesign}'

    def __init__(self,dicom_directory: str, slice_index: int, b_value = 50, b_values_list = [50,500,1200]):
        self.dicom_directory = dicom_directory
        self.slice_index = slice_index
        self.subdirectories_TRACEW = [
                                        self.dicom_directory+"/"+name
                                        for name in os.listdir(self.dicom_directory)
                                        if os.path.isdir(os.path.join(self.dicom_directory, name)) and name.endswith("TRACEW")
                                    ]
        self.b_values_list = b_values_list
        id_list = []
        for folder in self.subdirectories_TRACEW:
            id = os.path.basename(folder).split("_")[0]
            id_list.append(id)
        self.subdirectories = [self.dicom_directory+f"/{id}_b_value={b_value}" for id in id_list] # Used to select the ROIs


    def get_NIST_value(self,temperature, liquid:str):
        with open("/home/lbsc/IRM_diffusion/NIST_values_official.pkl", 'rb') as file:
            NIST_values_official = pkl.load(file)
        temperature_NIST = np.array([0,16,21,26.5,31,37])
        water_NIST_data = np.array([float(NIST_values_official["Water"][x][0]) for x in temperature_NIST])
        PVP50_NIST_data = np.array([float(NIST_values_official["PVP50"][x][0]) for x in temperature_NIST])
        PVP40_NIST_data = np.array([float(NIST_values_official["PVP40"][x][0]) for x in temperature_NIST])
        PVP30_NIST_data = np.array([float(NIST_values_official["PVP30"][x][0]) for x in temperature_NIST])
        PVP20_NIST_data = np.array([float(NIST_values_official["PVP20"][x][0]) for x in temperature_NIST])
        PVP10_NIST_data = np.array([float(NIST_values_official["PVP10"][x][0]) for x in temperature_NIST])

        water_coeff = np.polyfit(temperature_NIST,water_NIST_data,2)
        PVP50_coeff = np.polyfit(temperature_NIST,PVP50_NIST_data,2)
        PVP40_coeff = np.polyfit(temperature_NIST,PVP40_NIST_data,2)
        PVP30_coeff = np.polyfit(temperature_NIST,PVP30_NIST_data,2)
        PVP20_coeff = np.polyfit(temperature_NIST,PVP20_NIST_data,2)
        PVP10_coeff = np.polyfit(temperature_NIST,PVP10_NIST_data,2)

        water_value = water_coeff[0]*temperature**2+water_coeff[1]*temperature+water_coeff[2]
        PVP50_value = PVP50_coeff[0]*temperature**2+PVP50_coeff[1]*temperature+PVP50_coeff[2]
        PVP40_value = PVP40_coeff[0]*temperature**2+PVP40_coeff[1]*temperature+PVP40_coeff[2]
        PVP30_value = PVP30_coeff[0]*temperature**2+PVP30_coeff[1]*temperature+PVP30_coeff[2]
        PVP20_value = PVP20_coeff[0]*temperature**2+PVP20_coeff[1]*temperature+PVP20_coeff[2]
        PVP10_value = PVP10_coeff[0]*temperature**2+PVP10_coeff[1]*temperature+PVP10_coeff[2]

        water_uncert = 0.0035 +0.024*water_value
        PVP50_uncert = 0.0035 +0.024*PVP50_value
        PVP40_uncert = 0.0035 +0.024*PVP40_value
        PVP30_uncert = 0.0035 +0.024*PVP30_value
        PVP20_uncert = 0.0035 +0.024*PVP20_value
        PVP10_uncert = 0.0035 +0.024*PVP10_value

        dico_res = {"Water":(water_value,water_uncert),
                    "PVP50":(PVP50_value,PVP50_uncert),
                    "PVP40":(PVP40_value,PVP40_uncert),
                    "PVP30":(PVP30_value,PVP30_uncert),
                    "PVP20":(PVP20_value,PVP20_uncert),
                    "PVP10":(PVP10_value,PVP10_uncert)}
                
        return dico_res[liquid]


    def find_resolution_at_threshold_glob(self,mtf, freq, threshold):
            indices = np.where(mtf < threshold)[0]
            if len(indices) == 0:
                return None  # If no value found, return None
            
            idx = indices[0] #Gets lowest frequency under the specified threshold
            
            if idx == 0:
                return freq[0]
            
            x1, x2 = freq[idx - 1], freq[idx]
            y1, y2 = mtf[idx - 1], mtf[idx]
            
            resolution = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
            return resolution


    def load_dicom_series(self,directory_path: str):
        """
        Read all DCM files in the given directory, sort them by filename. Returns the pixel arrays and DCM tags

        Parameters
        ----------
        directory_path : str
            Path towards the directory containing the DICOM images.

        Returns
        ------
        images array : np.ndarray
            The array containing the gray values for all images in the specified folder.
        dico_metadata : list
            List containing the acquisition info of the images
        """
        dicom_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.dcm')]
        dicom_files.sort()
        
        images = []
        dicom_metadata = []
        for file in dicom_files:
            dicom = pydicom.dcmread(file)
            images.append(dicom.pixel_array)
            dicom_metadata.append(dicom)
        return np.array(images), dicom_metadata
    

    def show_images(self,block=True, time_lapse = 3, subdir = None):
        """
        Shows the images contained in the TRACEW.
        """
        if subdir is not None:
            subdirectory = subdir
        else:
            subdirectory = self.subdirectories_TRACEW[3]
        images, metadata_list = self.load_dicom_series(subdirectory)[0], self.load_dicom_series(subdirectory)[1]
        for x in range(len(images)):
            image = images[x]
            print("!!!!!!!!!!!!!")
            print(f"Minimum pixel value={np.min(np.array(image))}")
            print(f"Maximum pixel value={np.max(np.array(image))}")
            print("!!!!!!!!!!!!!")
            metadata = metadata_list[x]
            print(metadata)
            print("********************************")
            print(f"b={metadata[0x019,0x100c].value}") 
            print("Slice location")
            print(metadata[0x020,0x1041].value) 

            print(f"Size image={image.shape}")
            fig, ax = plt.subplots()
            ax.imshow(image, cmap='gray')
            plt.title(f"Slice number {x}, b={metadata[0x019,0x100c].value}")
            plt.savefig(f"/home/lbsc/IRM_diffusion/tempo/N1_slice{x}.png")
            plt.close()

    def b_value_split(self):
        for folder in self.subdirectories_TRACEW:
            id = os.path.basename(folder).split("_")[0]
            images, metadata_list = self.load_dicom_series(folder)[0], self.load_dicom_series(folder)[1]
            for b in self.b_values_list:
                b_folder_path = self.dicom_directory+f"/{id}_b_value={b}"
                if not os.path.exists(b_folder_path):
                    os.mkdir(b_folder_path)
                    print(f"Folder '{b_folder_path}' created.")
                else:
                    print(f"Folder '{b_folder_path}' already exists.")
            for x in range(len(images)):
                metadata = metadata_list[x]
                b_value = metadata[0x019,0x100c].value
                b_folder_path = os.path.join(self.dicom_directory, f"{id}_b_value={b_value}")
                output_filename = os.path.join(b_folder_path, f"{id}_{x:03d}.dcm")
                metadata.save_as(output_filename)
                print(f"Saved: {output_filename}")
                


    """Obsolete"""
    # def create_ADC_maps(self):
    #     ADC_folder_path = self.dicom_directory+f"/ADC_maps"
    #     if not os.path.exists(ADC_folder_path):
    #                 os.mkdir(ADC_folder_path)
    #                 print(f"Folder '{ADC_folder_path}' created.")
    #     couples = [
    #             (a, b)
    #             for a, b in itertools.permutations(self.b_values_list, 2)
    #             if b > a
    #                 ]
    #     for folder in self.subdirectories_TRACEW:
    #         id = os.path.basename(folder).split("_")[0]
    #         for small_b, big_b in couples:
    #             big_b_path = self.dicom_directory+f"/{id}_b_value={big_b}"
    #             small_b_path = self.dicom_directory+f"/{id}_b_value={small_b}"

    #             big_images, big_metadata_list = self.load_dicom_series(big_b_path)[0], self.load_dicom_series(big_b_path)[1]
    #             small_images = self.load_dicom_series(small_b_path)[0]


    #             ADC_maps = (1 / (big_b - small_b)) * np.log(small_images / big_images)

    #             np.save(self.dicom_directory+f"/ADC_maps/{id}_b{small_b}_b{big_b}",ADC_maps)

       
    #     for small_b, big_b in couples:
    #         tempo = []
    #         metadata_list = []
    #         for folder in self.subdirectories_TRACEW:
    #             id = os.path.basename(folder).split("_")[0]
    #             images = np.load(self.dicom_directory+f"/ADC_maps/{id}_b{small_b}_b{big_b}.npy")
    #             tempo.append(images)
    #         tempo = np.array(tempo)


    #         mean_ADC = np.mean(tempo,axis=0)
    #         std_ADC = np.std(tempo,axis=0)

    #         mean_ADC_path = self.dicom_directory+f"/ADC_maps/MEAN_b{small_b}_b{big_b}"
    #         std_ADC_path = self.dicom_directory+f"/ADC_maps/STD_b{small_b}_b{big_b}"
    #         np.save(mean_ADC_path,mean_ADC)
    #         print("Mean ADC map saved to "+mean_ADC_path)
    #         np.save(std_ADC_path ,std_ADC)
    #         print("STD ADC map saved to "+std_ADC_path)



    def create_ADC_maps_least_squares(self):
        ADC_folder_path = os.path.join(self.dicom_directory, "ADC_maps")
        if not os.path.exists(ADC_folder_path):
            os.mkdir(ADC_folder_path)
            print(f"Folder '{ADC_folder_path}' created.")

        b_values = np.array(self.b_values_list)

        # Compute ADC maps for each pass 
        for folder in self.subdirectories_TRACEW:
            id_ = os.path.basename(folder).split("_")[0]
            print(f"Processing {id_}...")

            # Load all DWI images for this ID (one per b-value)
            images_list = []
            for b in b_values:
                b_path = os.path.join(self.dicom_directory, f"{id_}_b_value={b}")
                images, _ = self.load_dicom_series(b_path)
                images_list.append(images.astype(np.float64))

            images_stack = np.stack(images_list, axis=0)  # shape: (n_b, z, y, x)


            # Take log of signal
            logS = np.log(images_stack)  # shape: (n_b, z, y, x)

            # Prepare linear regression: ln(S) = ln(S0) - b * ADC
            X = np.vstack([np.ones_like(b_values), -b_values]).T  # shape: (n_b, 2)
            # Precompute pseudoinverse for efficiency (since X is constant)
            X_pinv = np.linalg.pinv(X)

            # Reshape for vectorized regression: (n_b, n_vox)
            n_b, z, y, x = logS.shape
            logS_flat = logS.reshape(n_b, -1)  # (n_b, n_vox)

            # Solve least squares for each voxel: [ln(S0), ADC]
            coeffs = X_pinv @ logS_flat  # (2, n_vox)
            ADC_flat = coeffs[1, :]  # second row is ADC

            # Reshape back to image dimensions
            ADC_map = ADC_flat.reshape(z, y, x)

            # Save the ADC map
            np.save(os.path.join(ADC_folder_path, f"{id_}_ADC_map.npy"), ADC_map)
            print(f"Saved ADC map for {id_}.")

        # Compute mean and std ADC maps across all passes
        ADC_maps = []
        for folder in self.subdirectories_TRACEW:
            id_ = os.path.basename(folder).split("_")[0]
            path = os.path.join(ADC_folder_path, f"{id_}_ADC_map.npy")
            ADC_maps.append(np.load(path))

        ADC_maps = np.array(ADC_maps)
        mean_ADC = np.mean(ADC_maps, axis=0)
        std_ADC = np.std(ADC_maps, axis=0)

        mean_ADC_path = os.path.join(ADC_folder_path, "MEAN_ADC.npy")
        std_ADC_path = os.path.join(ADC_folder_path, "STD_ADC.npy")

        np.save(mean_ADC_path, mean_ADC)
        np.save(std_ADC_path, std_ADC)

        print(f"Mean ADC map saved to {mean_ADC_path}")
        print(f"STD ADC map saved to {std_ADC_path}")





    def get_pixel_size(self,dicom):
        """
        Extract the pixel size from the DCM metadata, used to compute MTF

        Parameters
        ----------
        dicom : pydicom.dataset.FileDataset
            Metadata of one image.

        Returns
        ------
        pixel_size : float
            The size of the pixels in the image in mm.
        """
        pixel_size = dicom.PixelSpacing[0]
        return pixel_size



    def get_seriesUID(self,dicom):
        """
        Extract the SeriesInstanceUID from the DCM metadata, used to identify the sequence

        Parameters
        ----------
        dicom : pydicom.dataset.FileDataset
            Metadata of one image.
        """
        
        return dicom.SeriesInstanceUID

    def get_field_strength(self,dicom):
        """
        Extract the field strength from the DCM metadata, used to identify the sequence

        Parameters
        ----------
        dicom : pydicom.dataset.FileDataset
            Metadata of one image.

        Returns
        ------
        dicom.MagneticFieldStrength : pydicom.valuerep.DSfloat
            Strength of the magnetic field in [T]
        """
        return dicom.MagneticFieldStrength if 'MagneticFieldStrength' in dicom else None
    

    def calculate_esf(self,image, start, end, width):
        """
        Computes the edge spread function (esf), given a start and end point. The width of the profile line is specified by the width parameter of the ROI.

        Parameters
        ----------
        image : np.ndarray
           Array containing the gray values of a specific image.
        start : tuple
            Coordinates of the first click when selecting the ROI.
        end : tuple
            Coordinates of the last click when selecting the ROI.
        width : float
            Indicates the width of the ROI. Along the y (x) axis when the ROI is selected horizontaly (verticaly).

        Returns
        ------
        profile : np.array
            The array containing the gray values of the average line along the ROI.
        """
        length_vector = np.array([end[0] - start[0], end[1] - start[1]])
        length_norm = np.hypot(length_vector[0], length_vector[1])
        if length_norm == 0:
            print("Warning: Length of the ROI is zero.")
            return np.zeros(1)
        unit_vector = length_vector / length_norm
        # Orthogonal direction to the ROI
        orthogonal_vector = np.array([-unit_vector[1], unit_vector[0]])
        profiles = []

        for i in range(-width // 2, width // 2 + 1):
            offset_start = np.array(start) + i * orthogonal_vector
            offset_end = np.array(end) + i * orthogonal_vector
            num_points = int(length_norm)
            x, y = np.linspace(offset_start[0], offset_end[0], num_points), np.linspace(offset_start[1], offset_end[1], num_points)
            profile_line = ndimage.map_coordinates(image, np.vstack((y, x)), order=1, mode='nearest')
            profiles.append(profile_line)
        
        profile = np.mean(profiles, axis=0)

        
        return profile

    def calculate_lsf(self,profile):
        """
        Computes the line spread function (lsf) from the given esf.

        Parameters
        ----------
        profile : np.array
           The array containing the gray values of the average line along the ROI.

        Returns
        ------
        Profile gradient : np.array
            The gradient of the profile, corresponding to the lsf.
        """
        return np.gradient(profile)

    def calculate_mtf(self,lsf, pixel_size):
        """
        Computes the modulation transfer function (mtf) from the given lsf.

        Parameters
        ----------
        lsf : np.array
           The lsf.
        pixel_size : float
            The size of the image pixel in [mm].

        Returns
        ------
        mtf : np.array
            The mtf values.
        freq : np.array
            The corresponding fequencies.
        """
        mtf = np.abs(fft(lsf))[:len(lsf) // 2]
        if np.max(mtf) == 0:
            return np.zeros(len(mtf)), np.zeros(len(mtf))
        mtf = mtf / np.max(mtf)
        freq = fftfreq(len(lsf), d=pixel_size)[:len(mtf)]
        return mtf, freq

    def calculate_resolution(self,mtf, freq):
        """
        Finds the frequencies associated with a 50% and 10% mtf value.

        Parameters
        ----------
        mtf : np.array
            The mtf values.
        freq : np.array
            The corresponding fequencies.

        Returns
        ------
        resolution_50 : float
            The frequency associated with a 50% value of mtf.
        resolution_10 : float
            The frequency associated with a 50% value of mtf.
        """
        def find_resolution_at_threshold(mtf, freq, threshold):
            indices = np.where(mtf < threshold)[0]
            if len(indices) == 0:
                return None  # If no value found, return None
            
            idx = indices[0] #Gets lowest frequency under the specified threshold
            
            if idx == 0:
                return freq[0]
            
            x1, x2 = freq[idx - 1], freq[idx]
            y1, y2 = mtf[idx - 1], mtf[idx]
            
            resolution = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
            return resolution

        resolution_50 = find_resolution_at_threshold(mtf, freq, 0.5)
        resolution_10 = find_resolution_at_threshold(mtf, freq, 0.1)
        return resolution_50, resolution_10

    horizontal_roi_norm = None
    vertical_roi_norm = None
    roi_selected = False
    mtf_curves = {}

    def onselect_horizontal(self,eclick, erelease):
        """
        Selects the horizontal ROI.
        """
        global horizontal_roi_norm, image_shape, horizontal_roi
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        horizontal_roi = [(x1, y1), (x2, y2)]

        print(f'Horizontal ROI: {horizontal_roi}')
        # Normalizes the ROI
        height, width = image_shape
        horizontal_roi_norm = [ (x1 / width, y1 / height), (x2 / width, y2 / height) ]

    def onselect_vertical(self,eclick, erelease):
        """
        Selects the vertical ROI.
        """
        global vertical_roi_norm, image_shape, vertical_roi
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        

        vertical_roi = [(x1, y1), (x2, y2)]
        print(f'Vertical ROI: {vertical_roi}')
        # Normalizes the ROI
        height, width = image_shape
        vertical_roi_norm = [ (x1 / width, y1 / height), (x2 / width, y2 / height) ]
    
    def onselect_object_ROI(self,eclick, erelease):
        """
        Selects the ROI of the object (signal).
        """
        global object_ROI, image_shape
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        

        object_ROI = [(x1, y1), (x2, y2)]
        print(f'Object ROI: {object_ROI}')


    def onselect_background_ROI(self,eclick, erelease):
        """
        Selects the ROI of the background (noise).
        """
        global background_ROI, image_shape
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        

        background_ROI = [(x1, y1), (x2, y2)]
        print(f'Background ROI: {background_ROI}')

    def onselect_water_1(self,eclick, erelease):
        """
        Selects the ROI of the background (noise).
        """
        global water_1_ROI, image_shape
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        

        water_1_ROI = [(x1, y1), (x2, y2)]
        print(f'Water 1 ROI: {water_1_ROI}')

    def onselect_water_2(self,eclick, erelease):
        """
        Selects the ROI of the background (noise).
        """
        global water_2_ROI, image_shape
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        

        water_2_ROI = [(x1, y1), (x2, y2)]
        print(f'Water 2 ROI: {water_2_ROI}')

    def onselect_PVP50_3(self,eclick, erelease):
        """
        Selects the ROI of the background (noise).
        """
        global PVP50_3_ROI, image_shape
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        

        PVP50_3_ROI = [(x1, y1), (x2, y2)]
        print(f'PVP50 3 ROI: {PVP50_3_ROI}')

    def onselect_PVP40_4(self,eclick, erelease):
        """
        Selects the ROI of the background (noise).
        """
        global PVP40_4_ROI, image_shape
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        

        PVP40_4_ROI = [(x1, y1), (x2, y2)]
        print(f'PVP40 4 ROI: {PVP40_4_ROI}')

    def onselect_PVP30_5(self,eclick, erelease):
        """
        Selects the ROI of the background (noise).
        """
        global PVP30_5_ROI, image_shape
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        

        PVP30_5_ROI = [(x1, y1), (x2, y2)]
        print(f'PVP30 5 ROI: {PVP30_5_ROI}')

    def onselect_PVP20_6(self,eclick, erelease):
        """
        Selects the ROI of the background (noise).
        """
        global PVP20_6_ROI, image_shape
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        

        PVP20_6_ROI = [(x1, y1), (x2, y2)]
        print(f'PVP20 6 ROI: {PVP20_6_ROI}')

    
    def onselect_PVP10_7(self,eclick, erelease):
        """
        Selects the ROI of the background (noise).
        """
        global PVP10_7_ROI, image_shape
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        

        PVP10_7_ROI = [(x1, y1), (x2, y2)]
        print(f'PVP10 7 ROI: {PVP10_7_ROI}')

    def onselect_PVP50_8(self,eclick, erelease):
        """
        Selects the ROI of the background (noise).
        """
        global PVP50_8_ROI, image_shape
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        

        PVP50_8_ROI = [(x1, y1), (x2, y2)]
        print(f'PVP50 8 ROI: {PVP50_8_ROI}')


    def onselect_PVP40_9(self,eclick, erelease):
        """
        Selects the ROI of the background (noise).
        """
        global PVP40_9_ROI, image_shape
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        

        PVP40_9_ROI = [(x1, y1), (x2, y2)]
        print(f'PVP40 9 ROI: {PVP40_9_ROI}')

    def onselect_PVP30_10(self,eclick, erelease):
        """
        Selects the ROI of the background (noise).
        """
        global PVP30_10_ROI, image_shape
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        

        PVP30_10_ROI = [(x1, y1), (x2, y2)]
        print(f'PVP30 10 ROI: {PVP30_10_ROI}')

    def onselect_PVP20_11(self,eclick, erelease):
        """
        Selects the ROI of the background (noise).
        """
        global PVP20_11_ROI, image_shape
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        

        PVP20_11_ROI = [(x1, y1), (x2, y2)]
        print(f'PVP20 11 ROI: {PVP20_11_ROI}')

    def onselect_PVP10_12(self,eclick, erelease):
        """
        Selects the ROI of the background (noise).
        """
        global PVP10_12_ROI, image_shape
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        

        PVP10_12_ROI = [(x1, y1), (x2, y2)]
        print(f'PVP10 12 ROI: {PVP10_12_ROI}')

    def onselect_water_13(self,eclick, erelease):
        """
        Selects the ROI of the background (noise).
        """
        global water_13_ROI, image_shape
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        

        water_13_ROI = [(x1, y1), (x2, y2)]
        print(f'Water 13 ROI: {water_13_ROI}')
        

    def on_button_click(self,event):
        """
        Creates buttons
        """
        global roi_selected
        roi_selected = True
        plt.close()

    def denormalize_roi(self,roi_norm, image_shape):
        """
        Denormalizes the ROI.
        """
        height, width = image_shape
        start = (roi_norm[0][0] * width, roi_norm[0][1] * height)
        end = (roi_norm[1][0] * width, roi_norm[1][1] * height)
        return [start, end]

    def process_sequence(self,dicom_directory, slice_index):
        """
        Extracts the mtf, frequencies and resolution for both horizontal and vertical ROI's.

        Parameters
        ----------
        dicom_directory : str
            Path towards the directory containing the DICOM images.
        slice_index : float
            Index of the chosen image.

        Returns
        ------
        horizontal_mtf : np.array
            The mtf of the horizontal ROI.
        horizontal_freq : np.array
            The freqencies of the horizontal ROI.
        horizontal_resolution : tuple
            Frequencies of 50% and 10% resolution.
        """
        images, dicom_metadata = self.load_dicom_series(dicom_directory)
        dicom = dicom_metadata[slice_index]
        pixel_size = self.get_pixel_size(dicom)
        sequence_name = dicom.SeriesDescription.replace("t2_tse_tra_", "").strip()
        image = images[slice_index]
        image_shape = image.shape  # (height, width)

        # Denormalize both ROI's
        horizontal_roi = self.denormalize_roi(horizontal_roi_norm, image_shape)
        vertical_roi = self.denormalize_roi(vertical_roi_norm, image_shape)

        # Compute the width of horizontal ROI
        horizontal_width = int(abs(horizontal_roi[1][1] - horizontal_roi[0][1]))
        # Compute the esf in horizontal direction
        horizontal_esf = self.calculate_esf(image, horizontal_roi[0], horizontal_roi[1], horizontal_width)
        # Compute the lsf in horizontal direction
        horizontal_lsf = self.calculate_lsf(horizontal_esf)
        # Compute the mtf in horizontal direction
        horizontal_mtf, horizontal_freq = self.calculate_mtf(horizontal_lsf, pixel_size)
        # Compute the limiting resolutions in horizontal direction
        horizontal_resolution = self.calculate_resolution(horizontal_mtf, horizontal_freq)

        # Compute the width of vertical ROI
        vertical_width = int(abs(vertical_roi[1][0] - vertical_roi[0][0]))
        # Compute the esf in vertical direction
        vertical_esf = self.calculate_esf(image, vertical_roi[0], vertical_roi[1], vertical_width)
        # Compute the lsf in vertical direction
        vertical_lsf = self.calculate_lsf(vertical_esf)
        # Compute the mtf in vertical direction
        vertical_mtf, vertical_freq = self.calculate_mtf(vertical_lsf, pixel_size)
        # Compute the limiting resolutions in vertical direction
        vertical_resolution = self.calculate_resolution(vertical_mtf, vertical_freq)

        return (horizontal_mtf, horizontal_freq, horizontal_resolution, vertical_mtf, vertical_freq, vertical_resolution, sequence_name)
    


    def ROI_select_MTF(self):
        """
        Selection of both horizontal and vertical ROI's.
        """
        global horizontal_roi_norm, vertical_roi_norm, roi_selected, image_shape, horizontal_roi, vertical_roi

        subdirectories = self.subdirectories

        if not subdirectories:
            print("No sub-folder found in DICOM repository.")
            return

        first_sequence_dir = subdirectories[0]
        images = self.load_dicom_series(first_sequence_dir)[0]
        image = images[self.slice_index]

        image_shape = image.shape  # (height, width)
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        plt.title("Select a horizontal ROI by clicking and sliding")

        horizontal_selector = RectangleSelector(ax, self.onselect_horizontal,
                                                useblit=False,
                                                button=[1], minspanx=5, minspany=5,
                                                spancoords='pixels', interactive=True)

        ax_button = plt.axes([0.8, 0.01, 0.1, 0.05])
        button = Button(ax_button, 'Validate')
        button.on_clicked(self.on_button_click)

        plt.show()

        while not roi_selected:
            plt.pause(0.1)

        roi_selected = False

        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        plt.title("Select a vertical ROI by clicking and sliding")

        vertical_selector = RectangleSelector(ax, self.onselect_vertical,
                                            useblit=False,
                                            button=[1], minspanx=5, minspany=5,
                                            spancoords='pixels', interactive=True)

        ax_button = plt.axes([0.8, 0.01, 0.1, 0.05])
        button = Button(ax_button, 'Validate')
        button.on_clicked(self.on_button_click)

        plt.show()

        while not roi_selected:
            plt.pause(0.1)

    def ROI_select_SNR_one_reg(self,shape="Rectangular"):
        """
        Selection of a single ROI's.

        Parameters
        ----------
        shape : str
            Shape of the ROI, either "Rectangular" or "circular".
        """
        if shape not in ["Rectangular", "Circular"]:
            raise Exception("ROI shape unavailable!")
        
        global roi_selected,image_shape, object_ROI

        subdirectories = self.subdirectories

        if not subdirectories:
            print("No sub-folder found in DICOM repository.")
            return

        first_sequence_dir = subdirectories[0]
        images = self.load_dicom_series(first_sequence_dir)[0]
        image = images[self.slice_index]
        fig, ax = plt.subplots()
        # ax.imshow(image[323:487,364:551], cmap='gray') #image[debut_y:fin_y,debut_x:fin_x]
        ax.imshow(image, cmap='gray')
        plt.title("Select a ROI containing the object by clicking and sliding")
        if shape == "Rectangular":
            object_selector = RectangleSelector(ax, self.onselect_object_ROI,
                                                    useblit=False,
                                                    button=[1], minspanx=5, minspany=5,
                                                    spancoords='pixels', interactive=True)
        else:
            object_selector = EllipseSelector(ax, self.onselect_object_ROI,
                                                    useblit=False,
                                                    button=[1], minspanx=5, minspany=5,
                                                    spancoords='pixels', interactive=True)

        ax_button = plt.axes([0.8, 0.01, 0.1, 0.05])
        button = Button(ax_button, 'Validate')
        button.on_clicked(self.on_button_click)

        plt.show()

        while not roi_selected:
            plt.pause(0.1)



    def ROI_select_13_reg(self, reference_png="/home/lbsc/IRM_diffusion/tubes_numérotés_phantom_25_oct.png",shape_ROI = "Rectangular"):
        global roi_selected, image_shape
        subdirectories = self.subdirectories
        if not subdirectories:
            print("No sub-folder found in DICOM repository.")
            return

        # Load DICOM image
        first_sequence_dir = subdirectories[0]
        images = self.load_dicom_series(first_sequence_dir)[0]
        image = images[self.slice_index]
        image_shape = image.shape

        # Load reference PNG
        ref_img = plt.imread(reference_png)

        # List of ROI to be selected (title + callback)
        roi_callbacks = [
            ("Select ROI Water_1", self.onselect_water_1),
            ("Select ROI Water_2", self.onselect_water_2),
            ("Select ROI PVP50_3", self.onselect_PVP50_3),
            ("Select ROI PVP40_4", self.onselect_PVP40_4),
            ("Select ROI PVP30_5", self.onselect_PVP30_5),
            ("Select ROI PVP20_6", self.onselect_PVP20_6),
            ("Select ROI PVP10_7", self.onselect_PVP10_7),
            ("Select ROI PVP50_8", self.onselect_PVP50_8),
            ("Select ROI PVP40_9", self.onselect_PVP40_9),
            ("Select ROI PVP30_10", self.onselect_PVP30_10),
            ("Select ROI PVP20_11", self.onselect_PVP20_11),
            ("Select ROI PVP10_12", self.onselect_PVP10_12),
            ("Select ROI Water_13", self.onselect_water_13)
        ]

        for title, callback in roi_callbacks:
            roi_selected = False

            # Create dual plot : DICOM + reference PNG
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

            # DICOM image
            ax1.imshow(image, cmap="gray")
            ax1.set_title(title)

            # RectangleSelector 
            if shape_ROI == "Rectangular":
                selector = RectangleSelector(ax1, callback,
                                            useblit=False,
                                            button=[1],
                                            minspanx=5, minspany=5,
                                            spancoords="pixels", interactive=True)
            else:
                selector = EllipseSelector(ax1, callback,
                                        useblit=False,
                                        button=[1],
                                        minspanx=5, minspany=5,
                                        spancoords="pixels", interactive=True)

            # Show reference PNG
            ax2.imshow(ref_img)
            ax2.axis("off")
            ax2.set_title("Reference")

            # Validate
            ax_button = plt.axes([0.45, 0.01, 0.1, 0.05])  # au centre sous la figure
            button = Button(ax_button, "Validate")
            button.on_clicked(self.on_button_click)

            plt.show()
            while not roi_selected:
                plt.pause(0.1)
        dico = {"Water_1": water_1_ROI,
                "Water_2": water_2_ROI,
                "PVP50_3": PVP50_3_ROI,
                "PVP40_4": PVP40_4_ROI,
                "PVP30_5": PVP30_5_ROI,
                "PVP20_6": PVP20_6_ROI,
                "PVP10_7": PVP10_7_ROI,
                "PVP50_8": PVP50_8_ROI,
                "PVP40_9": PVP40_9_ROI,
                "PVP30_10": PVP30_10_ROI,
                "PVP20_11": PVP20_11_ROI,
                "PVP10_12": PVP10_12_ROI,
                "Water_13": water_13_ROI,}
        print(dico)
        return dico


    

    def compute_SNR_region_single_image(self,shape_ROI : str):
        """
        Compute the value of the SNR by selecting one ROI of interest.

        Returns
        ------
        SNR : float
            The SNR  value associated with the object and background ROI's selected.
        """
        
            
        subdirectories = self.subdirectories_TRACEW
        first_sequence_dir = subdirectories[0]
        images = self.load_dicom_series(first_sequence_dir)[0]
        image = images[self.slice_index]
        # Signal of the object and background regions ROI
        object_signal, background_signal = [], []
        ellipse = []
        rectangle = []

        if shape_ROI == "Rectangular":

            self.ROI_select_SNR_one_reg(shape_ROI)
                

            for x in np.arange(min(object_ROI[0][1],object_ROI[1][1]), max(object_ROI[0][1],object_ROI[1][1]), 1):
                tempo = []
                for y in np.arange(min(object_ROI[0][0],object_ROI[1][0]), max(object_ROI[0][0],object_ROI[1][0]), 1):
                    object_signal.append(image[x][y])
                    tempo.append(image[x][y])
                rectangle.append(tempo)
            fig, ax = plt.subplots()
            ax.imshow(rectangle, cmap="gray")
            plt.title("Here is the ROI selected")
            plt.show()
            plt.close()

        if shape_ROI == "Circular":
            self.ROI_select_SNR_one_reg(shape_ROI)
            x1o, x2o, y1o, y2o = object_ROI[0][1], object_ROI[1][1], object_ROI[0][0], object_ROI[1][0]
            centero = ((x1o + x2o)/2, (y1o + y2o)/2)
            ao, bo = abs(centero[0]-object_ROI[0][1]), abs(centero[1]-object_ROI[0][0])
            for xo in np.arange(min(x1o,x2o), max(x1o,x2o), 1):
                tempo = []
                for yo in np.arange(min(y1o,y2o), max(y1o,y2o), 1):
                    if ((xo-centero[0])/ao)**2+((yo-centero[1])/bo)**2 <= 1:
                        object_signal.append(image[xo][yo])
                        tempo.append(image[xo][yo])
                    else:
                        tempo.append(0)
                ellipse.append(tempo)

    
            fig, ax = plt.subplots()
            ax.imshow(ellipse, cmap="gray")
            plt.title("Here is the ROI selected")
            plt.show()
            plt.close()

        object_signal_arr = np.array(object_signal)
        print("************************")
        print(len(object_signal_arr),len(object_signal))
        print("************************")



        # Compute the value of the numerator in the SNR
        numerator = np.linalg.norm(object_signal_arr.astype(np.float64))

        # Compute the average pixel value and the std of the background ROI
        average_denominator = sum(object_signal_arr.astype(np.float64))/len(object_signal_arr.astype(np.float64))
        std_denominator = np.linalg.norm(object_signal_arr.astype(np.float64)-average_denominator.astype(np.float64))/(len(object_signal_arr)-1)

        # Compute the SNR
        SNR = numerator/std_denominator
        print(f"The SNR value is {SNR}")
        return SNR
    

    def compute_SNR_region_multi_image(self,shape_ROI : str):
        """
        Compute the value of the SNR by selecting one or more ROI of interests.

        Returns
        ------
        SNR : float
            The SNR  value associated with the object and background ROI's selected.
        """
        
        self.ROI_select_SNR_one_reg(shape_ROI)
        subdirectories = self.subdirectories
        subdir_images = []
        for subdirectory in subdirectories:
            images = self.load_dicom_series(subdirectory)[0]
            image = images[self.slice_index]
            # Signal of the object and background regions ROI
            subdir_image = []


            if shape_ROI == "Rectangular":
                for x in np.arange(min(object_ROI[0][1],object_ROI[1][1]), max(object_ROI[0][1],object_ROI[1][1]), 1):
                    tempo = []
                    for y in np.arange(min(object_ROI[0][0],object_ROI[1][0]), max(object_ROI[0][0],object_ROI[1][0]), 1):
                        tempo.append(image[x][y])
                    subdir_image.append(tempo)

            

            if shape_ROI == "Circular":
                x1o, x2o, y1o, y2o = object_ROI[0][1], object_ROI[1][1], object_ROI[0][0], object_ROI[1][0]
                centero = ((x1o + x2o)/2, (y1o + y2o)/2)
                ao, bo = abs(centero[0]-object_ROI[0][1]), abs(centero[1]-object_ROI[0][0])
                for xo in np.arange(min(x1o,x2o), max(x1o,x2o), 1):
                    tempo = []
                    for yo in np.arange(min(y1o,y2o), max(y1o,y2o), 1):
                        if ((xo-centero[0])/ao)**2+((yo-centero[1])/bo)**2 <= 1:
                            tempo.append(image[xo][yo])
                        else:
                            tempo.append(0)
                    subdir_image.append(tempo)
            fig, ax = plt.subplots()
            ax.imshow(subdir_image, cmap="gray")
            plt.title("Here is the ROI selected")
            plt.show()
            plt.close()

            subdir_images.append(np.array(subdir_image))

        stack = np.stack(subdir_images, axis=0)
        signal_image = np.mean(stack, axis=0)
        non_zero_signal = signal_image[signal_image != 0]
        numerator = non_zero_signal.mean() if non_zero_signal.size > 0 else np.nan
        u_signal = np.std(non_zero_signal)/np.sqrt(non_zero_signal.size)


        noise_image = np.std(stack, axis=0)
        non_zero_noise = noise_image[noise_image != 0]
        denominator = non_zero_noise.mean() if non_zero_noise.size > 0 else np.nan
        u_noise = np.std(non_zero_noise)/np.sqrt(non_zero_noise.size)


        fig, ax = plt.subplots()
        ax.imshow(signal_image, cmap="gray")
        plt.title("Signal image")
        plt.show()
        plt.close()

        fig, ax = plt.subplots()
        ax.imshow(noise_image, cmap="gray")
        plt.title("Noise image")
        plt.show()
        plt.close()

        SNR = numerator/denominator
        uncert = SNR*np.sqrt((u_signal/numerator)**2+(u_noise/denominator)**2)
        print(f"SNR: {SNR}±{uncert}")
        return SNR, uncert

    def compute_qiba_metrics_13_reg(self, shape_ROI, reference_png="/home/lbsc/IRM_diffusion/tubes_numérotés_phantom_25_oct.png"):
        ROIS = self.ROI_select_13_reg(shape_ROI=shape_ROI,reference_png=reference_png)

        # subdirectories_SNR = self.subdirectories
        slice_index = self.slice_index 

        ###SNR computation###
        for b_value in self.b_values_list:
            id_list = []
            for folder in self.subdirectories_TRACEW:
                id = os.path.basename(folder).split("_")[0]
                id_list.append(id)
            subdirectories_SNR = [self.dicom_directory+f"/{id}_b_value={b_value}" for id in id_list]
            results_SNR = {}
            csv_path_SNR = os.path.join(self.dicom_directory, f"SNR_data/SNR_b={b_value}_slice={self.slice_index}.csv")
            csv_rows_SNR = []

            for roi_name, tube_ROI in ROIS.items():
                subdir_images = []

                for subdirectory in subdirectories_SNR:
                    images = self.load_dicom_series(subdirectory)[0]
                    image = images[slice_index]
                    subdir_image = []

                    if shape_ROI == "Rectangular":
                        for x in np.arange(min(tube_ROI[0][1], tube_ROI[1][1]),
                                        max(tube_ROI[0][1], tube_ROI[1][1]), 1):
                            tempo = []
                            for y in np.arange(min(tube_ROI[0][0], tube_ROI[1][0]),
                                            max(tube_ROI[0][0], tube_ROI[1][0]), 1):
                                tempo.append(image[x][y])
                            subdir_image.append(tempo)

                    if shape_ROI == "Circular":
                        x1o, x2o, y1o, y2o = tube_ROI[0][1], tube_ROI[1][1], tube_ROI[0][0], tube_ROI[1][0]
                        centero = ((x1o + x2o)/2, (y1o + y2o)/2)
                        ao, bo = abs(centero[0]-tube_ROI[0][1]), abs(centero[1]-tube_ROI[0][0])
                        for xo in np.arange(min(x1o, x2o), max(x1o, x2o), 1):
                            tempo = []
                            for yo in np.arange(min(y1o, y2o), max(y1o, y2o), 1):
                                if ((xo-centero[0])/ao)**2 + ((yo-centero[1])/bo)**2 <= 1:
                                    tempo.append(image[xo][yo])
                                else:
                                    tempo.append(0)
                            subdir_image.append(tempo)
                    number_of_pixels = np.count_nonzero(subdir_image)
                    subdir_images.append(np.array(subdir_image))

                stack = np.stack(subdir_images, axis=0)
                signal_image = np.mean(stack, axis=0)
                non_zero_signal = signal_image[signal_image != 0]
                numerator = non_zero_signal.mean() if non_zero_signal.size > 0 else np.nan
                u_signal = np.std(non_zero_signal)/np.sqrt(non_zero_signal.size)

                noise_image = np.std(stack, axis=0)
                non_zero_noise = noise_image[noise_image != 0]
                denominator = non_zero_noise.mean() if non_zero_noise.size > 0 else np.nan
                u_noise = np.std(non_zero_noise)/np.sqrt(non_zero_noise.size)

                SNR = numerator/denominator
                uncert = SNR*np.sqrt((u_signal/numerator)**2+(u_noise/denominator)**2)

                results_SNR[roi_name] = (SNR, uncert)

                # print(f"{roi_name}: SNR = {SNR:.2f} ± {uncert:.2f}")
            
                
                csv_rows_SNR.append({
                    "ROI": roi_name,
                    "b_values": f"{b_value}",
                    "SNR": results_SNR[roi_name][0],
                    "SNR_uncert": results_SNR[roi_name][1],
                    "ROI_pixel_number": number_of_pixels
                                    })
            path_folder_SNR = self.dicom_directory+"/SNR_data"
            if not os.path.exists(path_folder_SNR):
                os.mkdir(path_folder_SNR)
                print(f"Folder '{path_folder_SNR}' created.")

            with open(csv_path_SNR, mode="w", newline="") as csvfile_SNR:
                fieldnames_SNR = list(csv_rows_SNR[0].keys())
                writer = csv.DictWriter(csvfile_SNR, fieldnames=fieldnames_SNR)
                writer.writeheader()
                writer.writerows(csv_rows_SNR)
        print(f"Résultats SNR sauvegardés dans {csv_path_SNR}")

        ########################################################





        ###ADC stats computation###
        temp_obj = ADC_MRI_extractor(path=self.dicom_directory)
        temperature = temp_obj.get_temp()
        couples = [(a, b) for a, b in itertools.permutations(self.b_values_list, 2) if b > a]
        csv_path = os.path.join(self.dicom_directory, f"ADC_metrics/ADC_metrics_slice={self.slice_index}.csv")
        csv_rows = []
        for roi_name, tube_ROI in ROIS.items():
            liquid_name = roi_name.split("_")[0]
            ####Compute the ADC mean and std for each couple of b values###
            for small_b, big_b in couples:
                mean_ADC_image = np.load(self.dicom_directory+f"/ADC_maps/MEAN_b{small_b}_b{big_b}.npy")[self.slice_index]
                std_ADC_image = np.load(self.dicom_directory+f"/ADC_maps/STD_b{small_b}_b{big_b}.npy")[self.slice_index]
                

                mean_roi, std_roi = [], []
                if shape_ROI == "Rectangular":
                    for x in np.arange(min(tube_ROI[0][1], tube_ROI[1][1]),
                                    max(tube_ROI[0][1], tube_ROI[1][1]), 1):
                        tempo_mean = []
                        tempo_std = []
                        for y in np.arange(min(tube_ROI[0][0], tube_ROI[1][0]),
                                        max(tube_ROI[0][0], tube_ROI[1][0]), 1):
                            tempo_mean.append(mean_ADC_image[x][y])
                            tempo_std.append(std_ADC_image[x][y])
                        mean_roi.append(tempo_mean)
                        std_roi.append(tempo_std)
                    ADC_mean_value = np.mean(mean_roi)*1000 #mm^2/s
                    STD_mean_value = np.mean(std_roi)*1000 #mm^2/s
                    NIST_value = self.get_NIST_value(temperature=temperature, liquid=liquid_name)
                if shape_ROI == "Circular":
                    x1o, x2o, y1o, y2o = tube_ROI[0][1], tube_ROI[1][1], tube_ROI[0][0], tube_ROI[1][0]
                    centero = ((x1o + x2o)/2, (y1o + y2o)/2)
                    ao, bo = abs(centero[0]-tube_ROI[0][1]), abs(centero[1]-tube_ROI[0][0])
                    for xo in np.arange(min(x1o, x2o), max(x1o, x2o), 1):
                        tempo_mean = []
                        tempo_std = []
                        for yo in np.arange(min(y1o, y2o), max(y1o, y2o), 1):
                            if ((xo-centero[0])/ao)**2 + ((yo-centero[1])/bo)**2 <= 1:
                                tempo_mean.append(mean_ADC_image[xo][yo])
                                tempo_std.append(std_ADC_image[xo][yo])
                            else:
                                tempo_mean.append(0)
                                tempo_std.append(0)
                        mean_roi.append(tempo_mean)
                        std_roi.append(tempo_std)
                    mean_roi = np.array(mean_roi)
                    std_roi = np.array(std_roi)
                    mean_no_zero = mean_roi[mean_roi != 0]
                    std_no_zero = std_roi[std_roi != 0]

                    ADC_mean_value = np.mean(mean_no_zero)*1000 #mm^2/s
                    STD_mean_value = np.mean(std_no_zero)*1000 #mm^2/s
                
                number_of_pixels = np.count_nonzero(mean_roi)
                NIST_value, NIST_uncert = self.get_NIST_value(temperature=temperature, liquid=liquid_name)[0],self.get_NIST_value(temperature=temperature, liquid=liquid_name)[1]

                ADC_bias = ADC_mean_value - NIST_value
                ADC_bias_uncert = np.sqrt(STD_mean_value**2+ NIST_uncert**2)

                ADC_bias_percent = (ADC_bias/NIST_value)*100
                ADC_bias_percent_uncert = ADC_bias_percent*np.sqrt((ADC_bias_uncert/ADC_bias)**2+(NIST_uncert/NIST_value)**2)

                random_meas_perc = (STD_mean_value/ADC_mean_value)*100


                print(roi_name)
                print(f"ADC mean value: {ADC_mean_value} ± {STD_mean_value}")
                print(f"NIST value: {NIST_value} ± {NIST_uncert}")
                print(f"ADC bias: {ADC_bias} ± {ADC_bias_uncert}")
                print(f"ADC bias %: {ADC_bias_percent} ± {ADC_bias_percent_uncert}")

                csv_rows.append({
                "ROI": roi_name,
                "b_values": f"{small_b}-{big_b}",
                "ADC_mean": ADC_mean_value,
                "ADC_std": STD_mean_value,
                "NIST_value": NIST_value,
                "NIST_uncert": NIST_uncert,
                "ADC_bias": ADC_bias,
                "ADC_bias_uncert": ADC_bias_uncert,
                "ADC_bias_percent": ADC_bias_percent,
                "ADC_bias_percent_uncert": ADC_bias_percent_uncert,
                "Random_meas_%": random_meas_perc,
                "RC": 0,
                "wCV": 0,
                "ROI_pixel_number": number_of_pixels
                                })
                

            #### Compute the ADC mean and std for the total ADC map (least square fit) +RC+wCV ###
            id_list = []
            ### Compute RC ####
            ADC_per_pass_list = []
            for folder in self.subdirectories_TRACEW:
                id = os.path.basename(folder).split("_")[0]
                id_list.append(id)
            for id in id_list:
                images = np.load(self.dicom_directory+f"/ADC_maps/{id}_ADC_map.npy")
                image = images[self.slice_index]
                mean_roi = []
                if shape_ROI == "Rectangular":
                    for x in np.arange(min(tube_ROI[0][1], tube_ROI[1][1]),
                                    max(tube_ROI[0][1], tube_ROI[1][1]), 1):
                        tempo_mean = []
                        for y in np.arange(min(tube_ROI[0][0], tube_ROI[1][0]),
                                        max(tube_ROI[0][0], tube_ROI[1][0]), 1):
                            tempo_mean.append(image[x][y])
                        mean_roi.append(tempo_mean)
                    ADC_mean_value_pass = np.mean(mean_roi)*1000 #mm^2/s
                if shape_ROI == "Circular":
                    x1o, x2o, y1o, y2o = tube_ROI[0][1], tube_ROI[1][1], tube_ROI[0][0], tube_ROI[1][0]
                    centero = ((x1o + x2o)/2, (y1o + y2o)/2)
                    ao, bo = abs(centero[0]-tube_ROI[0][1]), abs(centero[1]-tube_ROI[0][0])
                    for xo in np.arange(min(x1o, x2o), max(x1o, x2o), 1):
                        tempo_mean = []
                        for yo in np.arange(min(y1o, y2o), max(y1o, y2o), 1):
                            if ((xo-centero[0])/ao)**2 + ((yo-centero[1])/bo)**2 <= 1:
                                tempo_mean.append(image[xo][yo])
                            else:
                                tempo_mean.append(0)
                        mean_roi.append(tempo_mean)
                    mean_roi = np.array(mean_roi)
                    mean_no_zero = mean_roi[mean_roi != 0]

                    ADC_mean_value_pass = np.mean(mean_no_zero)*1000 #mm^2/s

                ADC_per_pass_list.append(ADC_mean_value_pass)
            sig_w = np.std(ADC_per_pass_list)
            ################# Compute RC ###############################

            mean_ADC_image_tot = np.load(self.dicom_directory+f"/ADC_maps/MEAN_ADC.npy")[self.slice_index]
            std_ADC_image_tot = np.load(self.dicom_directory+f"/ADC_maps/STD_ADC.npy")[self.slice_index]
            mean_roi_tot, std_roi_tot = [], []
            if shape_ROI == "Rectangular":
                for x in np.arange(min(tube_ROI[0][1], tube_ROI[1][1]),
                                max(tube_ROI[0][1], tube_ROI[1][1]), 1):
                    tempo_mean_tot = []
                    tempo_std_tot = []
                    for y in np.arange(min(tube_ROI[0][0], tube_ROI[1][0]),
                                    max(tube_ROI[0][0], tube_ROI[1][0]), 1):
                        tempo_mean_tot.append(mean_ADC_image_tot[x][y])
                        tempo_std_tot.append(std_ADC_image_tot[x][y])
                    mean_roi_tot.append(tempo_mean_tot)
                    std_roi_tot.append(tempo_std_tot)
                ADC_mean_value_tot = np.mean(mean_roi_tot)*1000 #mm^2/s
                STD_mean_value_tot = np.mean(std_roi_tot)*1000 #mm^2/s
                NIST_value_tot = self.get_NIST_value(temperature=temperature, liquid=liquid_name)
            if shape_ROI == "Circular":
                x1o, x2o, y1o, y2o = tube_ROI[0][1], tube_ROI[1][1], tube_ROI[0][0], tube_ROI[1][0]
                centero = ((x1o + x2o)/2, (y1o + y2o)/2)
                ao, bo = abs(centero[0]-tube_ROI[0][1]), abs(centero[1]-tube_ROI[0][0])
                for xo in np.arange(min(x1o, x2o), max(x1o, x2o), 1):
                    tempo_mean_tot = []
                    tempo_std_tot = []
                    for yo in np.arange(min(y1o, y2o), max(y1o, y2o), 1):
                        if ((xo-centero[0])/ao)**2 + ((yo-centero[1])/bo)**2 <= 1:
                            tempo_mean_tot.append(mean_ADC_image_tot[xo][yo])
                            tempo_std_tot.append(std_ADC_image_tot[xo][yo])
                        else:
                            tempo_mean_tot.append(0)
                            tempo_std_tot.append(0)
                    mean_roi_tot.append(tempo_mean_tot)
                    std_roi_tot.append(tempo_std_tot)
                mean_roi_tot = np.array(mean_roi_tot)
                std_roi_tot = np.array(std_roi_tot)
                mean_no_zero_tot = mean_roi_tot[mean_roi_tot != 0]
                std_no_zero_tot = std_roi_tot[std_roi_tot != 0]

                ADC_mean_value_tot = np.mean(mean_no_zero_tot)*1000 #mm^2/s
                STD_mean_value_tot = np.mean(std_no_zero_tot)*1000 #mm^2/s

            NIST_value_tot, NIST_uncert_tot = self.get_NIST_value(temperature=temperature, liquid=liquid_name)[0],self.get_NIST_value(temperature=temperature, liquid=liquid_name)[1]

            ADC_bias_tot = ADC_mean_value_tot - NIST_value_tot
            ADC_bias_uncert_tot = np.sqrt(STD_mean_value_tot**2+ NIST_uncert_tot**2)

            ADC_bias_percent_tot = (ADC_bias_tot/NIST_value_tot)*100
            ADC_bias_percent_uncert_tot = ADC_bias_percent_tot*np.sqrt((ADC_bias_uncert_tot/ADC_bias_tot)**2+(NIST_uncert_tot/NIST_value_tot)**2)

            random_meas_perc_tot = (STD_mean_value_tot/ADC_mean_value_tot)*100


            print(roi_name)
            print(f"ADC mean value: {ADC_mean_value_tot} ± {STD_mean_value_tot}")
            print(f"NIST value: {NIST_value_tot} ± {NIST_uncert_tot}")
            print(f"ADC bias: {ADC_bias_tot} ± {ADC_bias_uncert_tot}")
            print(f"ADC bias %: {ADC_bias_percent_tot} ± {ADC_bias_percent_uncert_tot}")

            csv_rows.append({
            "ROI": roi_name,
            "b_values": f"TOTAL",
            "ADC_mean": ADC_mean_value_tot,
            "ADC_std": STD_mean_value_tot,
            "NIST_value": NIST_value_tot,
            "NIST_uncert": NIST_uncert_tot,
            "ADC_bias": ADC_bias_tot,
            "ADC_bias_uncert": ADC_bias_uncert_tot,
            "ADC_bias_percent": ADC_bias_percent_tot,
            "ADC_bias_percent_uncert": ADC_bias_percent_uncert_tot,
            "Random_meas_%": random_meas_perc_tot,
            "RC": 2.77*sig_w,
            "wCV": (sig_w/ADC_mean_value_tot)*100,
            "ROI_pixel_number": number_of_pixels})
                
        path_folder_ADC = self.dicom_directory+"/ADC_metrics"
        if not os.path.exists(path_folder_ADC):
            os.mkdir(path_folder_ADC)
            print(f"Folder '{path_folder_ADC}' created.")
        with open(csv_path, mode="w", newline="") as csvfile:
            fieldnames = list(csv_rows[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"Résultats sauvegardés dans {csv_path}")

                 
        

    def compute_MTF(self, save_to_csv=True):
        """
        Plots the value of the MTF as a function of spatial frequency.
        """
        acqu_name = self.dicom_directory.split("/")[-1]
        results_path = self.dicom_directory+f"/MTF_data"
        if not os.path.exists(results_path):
            os.makedirs(results_path, exist_ok=True)
            print(f"Folder '{results_path}' created.")
        self.ROI_select_MTF()


        for b_value in self.b_values_list:
            id_list = []
            for folder in self.subdirectories_TRACEW:
                id = os.path.basename(folder).split("_")[0]
                id_list.append(id)
            subdirectories = [self.dicom_directory+f"/{id}_b_value={b_value}" for id in id_list]

            

            if not subdirectories:
                print("No sub-folder found in DICOM repository.")
                return
            first_sequence_dir = subdirectories[0]
            images,dicom_metadata = self.load_dicom_series(first_sequence_dir)
            dicom = dicom_metadata[self.slice_index]
            seriesUID = self.get_seriesUID(dicom)
            field_strength = self.get_field_strength(dicom)


            

            if horizontal_roi_norm and vertical_roi_norm:
                all_horizontal_mtf = []
                all_horizontal_freq = []
                all_horizontal_resolutions = []
                all_vertical_mtf = []
                all_vertical_freq = []
                all_vertical_resolutions = []
                sequence_names = []
                field_strengths = []
                seriesUIDs = []

                if save_to_csv:
                    # MTF results file
                    csvfile_results = open(f"{self.dicom_directory}/MTF_data/{acqu_name}_b{b_value}_sliceindex{self.slice_index}_MTF_results.csv", 'w', newline='')
                    fieldnames_results = ['Sequence Name', 'MTF50% Horizontal', 'MTF10% Horizontal', 'MTF50% Vertical', 'MTF10% Vertical', 'FieldStrength', 'SeriesInstance UID']
                    writer_results = csv.DictWriter(csvfile_results, fieldnames=fieldnames_results)
                    writer_results.writeheader()

                for subdirectory in subdirectories:
                    print(f"Traitement de la séquence : {subdirectory}")
                    image,dicom_metadata =self.load_dicom_series(subdirectory)[0], self.load_dicom_series(subdirectory)[1]
                    dicom = dicom_metadata[self.slice_index]
                    sequence_name = os.path.basename(subdirectory)
                    seriesUID = self.get_seriesUID(dicom)
                    curr_field_strength = self.get_field_strength(dicom)



                    try:
                        (horizontal_mtf, horizontal_freq, horizontal_resolution,
                        vertical_mtf, vertical_freq, vertical_resolution, _) = self.process_sequence(subdirectory, self.slice_index)

                        
                        self.mtf_curves[sequence_name] = {}
                        
                        self.mtf_curves[sequence_name] = {
                            'horizontal': {
                                'frequency': horizontal_freq.tolist(),
                                'mtf': horizontal_mtf.tolist()
                            },
                            'vertical': {
                                'frequency': vertical_freq.tolist(),
                                'mtf': vertical_mtf.tolist()
                            }
                        }

                        all_horizontal_mtf.append(horizontal_mtf)
                        all_horizontal_freq.append(horizontal_freq)
                        all_horizontal_resolutions.append(horizontal_resolution)
                        all_vertical_mtf.append(vertical_mtf)
                        all_vertical_freq.append(vertical_freq)
                        all_vertical_resolutions.append(vertical_resolution)
                        sequence_names.append(sequence_name)
                        field_strengths.append(curr_field_strength)
                        seriesUIDs.append(seriesUID)

                        if save_to_csv:
                            res_50_h, res_10_h = horizontal_resolution
                            res_50_v, res_10_v = vertical_resolution
                            writer_results.writerow({
                                'Sequence Name': sequence_name,
                                'MTF50% Horizontal': res_50_h,
                                'MTF10% Horizontal': res_10_h,
                                'MTF50% Vertical': res_50_v,
                                'MTF10% Vertical': res_10_v,
                                'FieldStrength': field_strength,
                                'SeriesInstance UID': seriesUID
                            })

                    except Exception as e:
                        print(f"Mistake occured during treatment of sequence {sequence_name}: {e}")
                        continue
                stack_hori = np.stack (all_horizontal_mtf, axis=0)
                mean_hori = np.mean(stack_hori, axis=0)
                uncert_hori = np.std(stack_hori, axis=0)/np.sqrt(len(stack_hori))
                stack_verti = np.stack (all_vertical_mtf, axis=0)
                mean_verti = np.mean(stack_verti, axis=0)
                uncert_verti = np.std(stack_verti, axis=0)/np.sqrt(len(stack_verti))
                res_50_mean_h = self.find_resolution_at_threshold_glob(mean_hori,all_horizontal_freq[0],0.5*max(mean_hori))
                res_10_mean_h = self.find_resolution_at_threshold_glob(mean_hori,all_horizontal_freq[0],0.1*max(mean_hori))
                res_50_mean_v = self.find_resolution_at_threshold_glob(mean_verti,all_vertical_freq[0],0.5*max(mean_verti))
                res_10_mean_v = self.find_resolution_at_threshold_glob(mean_verti,all_vertical_freq[0],0.1*max(mean_verti))
                self.mtf_curves["MEAN"] = {
                            'horizontal': {
                                'frequency': all_horizontal_freq[0].tolist(),
                                'mtf': mean_hori.tolist()
                            },
                            'vertical': {
                                'frequency': all_vertical_freq[0].tolist(),
                                'mtf': mean_verti.tolist()
                            }}
                self.mtf_curves["Uncert"] = {
                            'horizontal': {
                                'frequency': all_horizontal_freq[0].tolist(),
                                'mtf': uncert_hori.tolist()
                            },
                            'vertical': {
                                'frequency': all_vertical_freq[0].tolist(),
                                'mtf': uncert_verti.tolist()
                            }}
                # Create folder if not exist
                subdir_name = os.path.basename(self.dicom_directory)
                # results_path = os.path.join("Results", f"{field_strength}T", "MTF", subdir_name)

                with open(os.path.join(results_path, f"{acqu_name}_b{b_value}_sliceindex{self.slice_index}_MTF_curves.pkl"), 'wb') as f:
                    pkl.dump(self.mtf_curves, f)
                
                if save_to_csv:
                    csvfile_results.close()

                plt.figure(figsize=(12, 8))
                
                sorted_indices = sorted(range(len(sequence_names)), key=lambda i: int(''.join(filter(str.isdigit, sequence_names[i]))))
                plt.suptitle(f"B value: {b_value}, Slice index: {self.slice_index}")
                plt.subplot(2, 1, 1)
                for i in sorted_indices:
                    res_50, res_10 = all_horizontal_resolutions[i]
                    if res_50 is not None and res_10 is not None:
                        label = f"Sequence {sequence_names[i].split("_")[0]} (MTF$_{{50\\%}}$: {res_50:.3f} cycles/mm, MTF$_{{10\\%}}$: {res_10:.3f} cycles/mm)"
                    else:
                        label = f"Sequence {sequence_names[i].split("_")[0]} (MTF$_{{50\\%}}$: N/A, MTF$_{{10\\%}}$: N/A)"
                    plt.plot(all_horizontal_freq[i], all_horizontal_mtf[i], label=label)

                plt.title(r'Frequency Modulation Transfer Function')
                plt.legend(fontsize=10)
                
                plt.subplot(2, 1, 2)
                for i in sorted_indices:
                    res_50, res_10 = all_vertical_resolutions[i]
                    if res_50 is not None and res_10 is not None:
                        label = f"Sequence {sequence_names[i].split("_")[0]} (MTF$_{{50\\%}}$: {res_50:.3f} cycles/mm, MTF$_{{10\\%}}$: {res_10:.3f} cycles/mm)"
                    else:
                        label = f"Sequence {sequence_names[i].split("_")[0]} (MTF$_{{50\\%}}$: N/A, MTF$_{{10\\%}}$: N/A)"
                    plt.plot(all_vertical_freq[i], all_vertical_mtf[i], label=label)

                    
                plt.title(r'Phase Modulation Transfer Function')
                plt.xlabel('Spatial Frequency (cycles/mm)')
                plt.legend(fontsize=10)
                plt.tight_layout()
                plt.savefig(results_path+f"/{acqu_name}_all_curves_b{b_value}_sliceindex{self.slice_index}.pdf")
                plt.show()
                plt.close()
                


                plt.subplot(2, 1, 1)
                plt.suptitle(f"B value: {b_value}, Slice index: {self.slice_index}")
                if res_50_mean_h is not None and res_10_mean_h is not None:
                    label_mean_h = f"MTF$_{{50\\%}}$: {res_50_mean_h:.3f} cycles/mm, MTF$_{{10\\%}}$: {res_10_mean_h:.3f} cycles/mm"
                else:
                    label_mean_h = f"MTF$_{{50\\%}}$: N/A, MTF$_{{10\\%}}$: N/A"
                plt.plot(all_horizontal_freq[0], mean_hori, label = label_mean_h, color = "blue")
                plt.title(r'Frequency Modulation Transfer Function')
                plt.fill_between(all_horizontal_freq[0],
                    mean_hori - uncert_hori,
                    mean_hori + uncert_hori,
                    color='blue', alpha=0.3)
                plt.legend(fontsize=10)

                plt.subplot(2, 1, 2)
                if res_50_mean_v is not None and res_10_mean_v is not None:
                    label_mean_v = f"MTF$_{{50\\%}}$: {res_50_mean_v:.3f} cycles/mm, MTF$_{{10\\%}}$: {res_10_mean_v:.3f} cycles/mm"
                else:
                    label_mean_v = f"MTF$_{{50\\%}}$: N/A, MTF$_{{10\\%}}$: N/A"
                plt.plot(all_vertical_freq[0], mean_verti, label = label_mean_v, color = "orange")
                plt.title(r'Phase Modulation Transfer Function')
                plt.xlabel('Spatial Frequency (cycles/mm)')
                plt.fill_between(all_vertical_freq[0],
                    mean_verti - uncert_verti,
                    mean_verti + uncert_verti,
                    color='orange', alpha=0.3)
                plt.legend(fontsize=10)
                plt.tight_layout()
                plt.savefig(results_path+f"/{acqu_name}_mean_curves_b{b_value}_sliceindex{self.slice_index}.pdf")
                plt.show()
                plt.close()
            else:
                print("No ROI selected")
                plt.close()

    






