"""
    @file:              DicomImage_class.py
    @Author:            Philippe Dionne et Louis-Bernard St-Cyr

    @Creation Date:     08/2025
    @Last modification: 09/2025

    @Description:       A modified version of the MTF code of Philippe Dionne, with the added options of computing the SNR in pixels and frequency.
"""


import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.fft import fft, fftfreq
from numpy.fft import fft2, ifft2, fftshift
from matplotlib.widgets import RectangleSelector, Button
import csv
import pickle as pkl
from pylinac.core.nps import noise_power_spectrum_1d
from scipy.interpolate import interp1d

class DicomImage:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = r'\usepackage[charter]{mathdesign}'

    def __init__(self,dicom_directory: str, slice_index: int, save_to_csv=False):
        self.dicom_directory = dicom_directory
        self.slice_index = slice_index
        self.save_to_csv = save_to_csv


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

        Returns
        ------
        pixel_size : pydicom.uid.UID
            UID of the image
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

        subdirectories = [os.path.join(self.dicom_directory, d) for d in os.listdir(self.dicom_directory) if os.path.isdir(os.path.join(self.dicom_directory, d))]

        if not subdirectories:
            print("No sub-folder found in DICOM repository.")
            return

        first_sequence_dir = subdirectories[0]
        images = self.load_dicom_series(first_sequence_dir)[0]
        image = images[slice_index]

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

    def ROI_select_SNR(self):
        """
        Selection of both object and background ROI's.
        """
        global object_ROI, background_ROI, roi_selected,image_shape

        subdirectories = [os.path.join(self.dicom_directory, d) for d in os.listdir(self.dicom_directory) if os.path.isdir(os.path.join(self.dicom_directory, d))]

        if not subdirectories:
            print("No sub-folder found in DICOM repository.")
            return

        first_sequence_dir = subdirectories[0]
        images = self.load_dicom_series(first_sequence_dir)[0]
        image = images[slice_index]
        fig, ax = plt.subplots()
        # ax.imshow(image[323:487,364:551], cmap='gray') #image[debut_y:fin_y,debut_x:fin_x]
        ax.imshow(image, cmap='gray')
        plt.title("Select a ROI containing the object by clicking and sliding")

        object_selector = RectangleSelector(ax, self.onselect_object_ROI,
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
        plt.title("Select a ROI containing the background by clicking and sliding")

        background_selector = RectangleSelector(ax, self.onselect_background_ROI,
                                            useblit=False,
                                            button=[1], minspanx=5, minspany=5,
                                            spancoords='pixels', interactive=True)

        ax_button = plt.axes([0.8, 0.01, 0.1, 0.05])
        button = Button(ax_button, 'Validate')
        button.on_clicked(self.on_button_click)

        plt.show()

        while not roi_selected:
            plt.pause(0.1)

    def compute_SNR_pixel(self):
        """
        Compute the value of the SNR by selecting the ROI's of the object of interest and the background

        Returns
        ------
        SNR : float
            The SNR  value associated with the object and background ROI's selected.
        """
        # Select ROI's
        self.ROI_select_SNR()

        subdirectories = [os.path.join(self.dicom_directory, d) for d in os.listdir(self.dicom_directory) if os.path.isdir(os.path.join(self.dicom_directory, d))]
        first_sequence_dir = subdirectories[0]
        images = self.load_dicom_series(first_sequence_dir)[0]
        image = images[slice_index]
        # Signal of the object and background regions ROI
        object_signal, background_signal = [], []

        for x in np.arange(min(object_ROI[0][1],object_ROI[1][1])):
            for y in np.arange(min(object_ROI[0][0],object_ROI[1][0])):
                object_signal.append(image[x][y])
        object_signal_arr = np.array(object_signal)

        for x in np.arange(min(background_ROI[0][1],background_ROI[1][1])):
            for y in np.arange(min(background_ROI[0][0],background_ROI[1][0])):
                background_signal.append(image[x][y])
        background_signal_arr = np.array(background_signal)

        # Compute the value of the numerator in the SNR
        numerator = np.linalg.norm(object_signal_arr.astype(np.float64))

        # Compute the average pixel value and the std of the background ROI
        average_background = sum(background_signal_arr)/len(background_signal_arr)
        std_background = np.linalg.norm(background_signal_arr.astype(np.float64)-average_background.astype(np.float64))/(len(background_signal_arr)-1)

        # Compute the SNR
        SNR = numerator/std_background
        print(f"The SNR value is {SNR}")
        return SNR
    def autocovariance_2d(self,image):
        """
        Computes the 2D autocovariance function of an image.

        Parameters
        ----------
        image : np.ndarray
            2D image (grayscale).

        Returns
        ------
        autocov : np.ndarray
            Autocovariance of thd 2D image.
        """

        img = image.astype(float)

        # Center the image
        img_centered = img - np.mean(img)

        # Compute de 2D FFT (Wiener–Khinchin theorem)
        f_img = fft2(img_centered)
        #Power spectral density
        psd = np.abs(f_img) ** 2
        autocov = np.real(ifft2(psd))

        # Shifts the zero at the center of the image
        autocov = fftshift(autocov)

        # Normalisation 
        autocov /= autocov.max()

        return autocov
    

    def noise_power_spectra(self):
        """
        Computes the noise power spectra (NPS).

        Returns
        ------
        nps1d_h : np.array
            1D NPS of the horizontal image.
        freq_h : np.array
            Frequencies of the horizontal image.
        nps1d_v : np.array
            1D NPS of the vertical image.
        freq_v : np.array
            Frequencies of the vertical image.
        mean_horizontal_val : float
            Mean pixel value of the horizontal ROI.
        mean_vertical_val : float
            Mean pixel value of the vertical ROI.
        """

        subdirectories = [os.path.join(self.dicom_directory, d) for d in os.listdir(self.dicom_directory) if os.path.isdir(os.path.join(self.dicom_directory, d))]

        if not subdirectories:
            print("No sub-folder found in DICOM repository.")
            return

        first_sequence_dir = subdirectories[0]

        image, dicom_metadata = self.load_dicom_series(first_sequence_dir)[0][self.slice_index], self.load_dicom_series(first_sequence_dir)[1][self.slice_index]

        pixel_size = self.get_pixel_size(dicom_metadata)

        print(f"PIXEL SIZE: {pixel_size}")

        #image[debut_y:fin_y,debut_x:fin_x]
        hy_start, hy_end = min(horizontal_roi[0][1],horizontal_roi[1][1]), max(horizontal_roi[0][1],horizontal_roi[1][1])

        hx_start, hx_end = min(horizontal_roi[0][0],horizontal_roi[1][0]), max(horizontal_roi[0][0],horizontal_roi[1][0])

        vy_start, vy_end = min(vertical_roi[0][1],vertical_roi[1][1]), max(vertical_roi[0][1],vertical_roi[1][1])

        vx_start, vx_end = min(vertical_roi[0][0],vertical_roi[1][0]), max(vertical_roi[0][0],vertical_roi[1][0])
        
        #Extracts the horizontal and vertical images selected from the ROI's.
        horizontal_image = image[hy_start:hy_end,hx_start:hx_end]
        vertical_image = image[vy_start:vy_end,vx_start:vx_end]

        #Compute the mean pixel value of each ROI.
        mean_horizontal_val = np.mean(horizontal_image)
        mean_vertical_val = np.mean(vertical_image)

        #Compute the autocovariance function of the horizontal ROI.
        autocovh = self.autocovariance_2d(horizontal_image)
        fft_autocovh = fft2(autocovh)
        fft_autocovh_shifted = fftshift(fft_autocovh)
        #Compute the 2D nps of the horizontal ROI.
        nps2d_h = np.abs(fft_autocovh_shifted)

        #Compute the autocovariance function of the vertical ROI.
        autocovv = self.autocovariance_2d(vertical_image)
        fft_autocovv = fft2(autocovv)
        fft_autocovv_shifted = fftshift(fft_autocovv)
        #Compute the 2D nps of the vertical ROI.
        nps2d_v = np.abs(fft_autocovv_shifted)

        #Compute the 2D nps of the horizontal ROI.
        nps1d_h = noise_power_spectrum_1d(spectrum_2d=nps2d_h)
        #Compute the 1D nps of the horizontal ROI.
        nps1d_v = noise_power_spectrum_1d(spectrum_2d=nps2d_v)

        #Compute frencies up to Nyquist frequency
        freq_h = np.linspace(0, 0.5/pixel_size, len(nps1d_h))
        freq_v = np.linspace(0, 0.5/pixel_size, len(nps1d_v))

                
        return nps1d_h, freq_h, nps1d_v, freq_v, mean_horizontal_val, mean_vertical_val

    def plot_SNR_frequency(self, log=False):
        """
        Plots the value of the SNR as a function of spatial frequency, by using the MTF ROI's.

        Parameters
        ----------
        log : bool
            Set to True if you want the axis to be in log scale.

        Returns
        ------
        SNR_h : np.array
            SNR of the horizontal ROI.
        SNR_v : np.array
            SNR of the vertical ROI.
        """

        #Select the horizontal and vertical ROI's.
        self.ROI_select_MTF()

        nps1d_h, freq_h_nps, nps1d_v, freq_v_nps, mean_horizontal_val, mean_vertical_val= self.noise_power_spectra()



        subdirectories = [os.path.join(self.dicom_directory, d) for d in os.listdir(self.dicom_directory) if os.path.isdir(os.path.join(self.dicom_directory, d))]

        if horizontal_roi_norm and vertical_roi_norm:
            all_horizontal_mtf = []
            all_horizontal_freq = []
            all_vertical_mtf = []
            all_vertical_freq = []
            sequence_names = []

            for subdirectory in subdirectories:
                print(f"Treating sequence : {subdirectory}")
                dicom_metadata = self.load_dicom_series(subdirectory)[1]
                dicom = dicom_metadata[slice_index]
                sequence_name = dicom.SeriesDescription.replace("t2_tse_tra_", "").strip()

                try:
                    (horizontal_mtf, horizontal_freq, horizontal_resolution,
                    vertical_mtf, vertical_freq, vertical_resolution, _) = self.process_sequence(subdirectory, slice_index)
                    all_horizontal_mtf.append(horizontal_mtf)
                    all_horizontal_freq.append(horizontal_freq)
                    all_vertical_mtf.append(vertical_mtf)
                    all_vertical_freq.append(vertical_freq)
                    sequence_names.append(sequence_name)

                except Exception as e:
                    print(f"Mistake occured during treatment of sequence {sequence_name}: {e}")
                    continue


            sorted_indices = sorted(range(len(sequence_names)), key=lambda i: int(''.join(filter(str.isdigit, sequence_names[i]))))
                

            
            plt.figure(figsize=(12, 8))
            for i in sorted_indices:
                nps1d_h_on_mtf = interp1d(freq_h_nps, nps1d_h, kind='linear', fill_value='extrapolate')(all_horizontal_freq[i])
                nps1d_v_on_mtf = interp1d(freq_v_nps, nps1d_v, kind='linear', fill_value='extrapolate')(all_vertical_freq[i])
                SNR_h = (all_horizontal_mtf[i]/nps1d_h_on_mtf)*mean_horizontal_val
                SNR_v = (all_vertical_mtf[i]/nps1d_v_on_mtf)*mean_vertical_val
                plt.subplot(2, 1, 1)
                plt.plot(all_horizontal_freq[i], SNR_h,label="Horizontal SNR")
                plt.title(r'Horizontal ROI')
                if log:
                    plt.xscale('log')    # Axe x en log
                    plt.yscale('log')
                    plt.ylabel("log[SNR]")
                else:
                    plt.ylabel("SNR")
                plt.legend()
                
                plt.subplot(2, 1, 2)
                plt.plot(all_vertical_freq[i], SNR_v,label="Vertical SNR")    
                plt.title(r'Vertical ROI')
                if log:
                    plt.xscale('log')    # Axe x en log
                    plt.yscale('log')
                    plt.xlabel('log [Spatial Frequency (cycles/mm)]')
                    plt.ylabel("log[SNR]")
                else:
                    plt.xlabel('Spatial Frequency (cycles/mm)')
                    plt.ylabel("SNR")
                plt.legend()
            
            plt.tight_layout()
            plt.show()
            plt.close()


        else:
            print("No ROI selected")
            plt.close()
        return SNR_h, SNR_v
        

    def plot_MTF(self):
        """
        Plots the value of the MTF as a function of spatial frequency.
        """

        #ROI selection
        self.ROI_select_MTF()
        

        subdirectories = [os.path.join(self.dicom_directory, d) for d in os.listdir(self.dicom_directory) if os.path.isdir(os.path.join(self.dicom_directory, d))]

        if not subdirectories:
            print("No sub-folder found in DICOM repository.")
            return

        first_sequence_dir = subdirectories[0]
        images, dicom_metadata = self.load_dicom_series(first_sequence_dir)
        dicom = dicom_metadata[slice_index]
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

            if self.save_to_csv:
                # MTF results file
                csvfile_results = open(f"{dicom_directory}_MTF_results.csv", 'w', newline='')
                fieldnames_results = ['Sequence Name', 'MTF50% Horizontal', 'MTF10% Horizontal', 'MTF50% Vertical', 'MTF10% Vertical', 'FieldStrength', 'SeriesInstance UID']
                writer_results = csv.DictWriter(csvfile_results, fieldnames=fieldnames_results)
                writer_results.writeheader()

            for subdirectory in subdirectories:
                print(f"Traitement de la séquence : {subdirectory}")
                dicom_metadata = self.load_dicom_series(subdirectory)[1]
                dicom = dicom_metadata[slice_index]
                sequence_name = dicom.SeriesDescription.replace("t2_tse_tra_", "").strip()
                seriesUID = self.get_seriesUID(dicom)
                curr_field_strength = self.get_field_strength(dicom)



                try:
                    (horizontal_mtf, horizontal_freq, horizontal_resolution,
                    vertical_mtf, vertical_freq, vertical_resolution, _) = self.process_sequence(subdirectory, slice_index)

                    if sequence_name not in self.mtf_curves:
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

                    if self.save_to_csv:
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

            # Création du répertoire principal s'il n'existe pas
            subdir_name = os.path.basename(dicom_directory)
            results_path = os.path.join("Results", f"{field_strength}T", "MTF", subdir_name)
            os.makedirs(results_path, exist_ok=True)
            with open(os.path.join(results_path, f"{subdir_name}_MTF_curves.pkl"), 'wb') as f:
                pkl.dump(self.mtf_curves, f)
            
            if self.save_to_csv:
                csvfile_results.close()

            plt.figure(figsize=(12, 8))
            
            sorted_indices = sorted(range(len(sequence_names)), key=lambda i: int(''.join(filter(str.isdigit, sequence_names[i]))))
            
            plt.subplot(2, 1, 1)
            for i in sorted_indices:
                res_50, res_10 = all_horizontal_resolutions[i]
                if res_50 is not None and res_10 is not None:
                    label = f"{sequence_names[i]} (MTF$_{{50\\%}}$: {res_50:.3f} cycles/mm, MTF$_{{10\\%}}$: {res_10:.3f} cycles/mm)"
                else:
                    label = f"{sequence_names[i]} (MTF$_{{50\\%}}$: N/A, MTF$_{{10\\%}}$: N/A)"
                plt.plot(all_horizontal_freq[i], all_horizontal_mtf[i], label=label)

            plt.title(r'Frequency Modulation Transfer Function')
            plt.legend()
            
            plt.subplot(2, 1, 2)
            for i in sorted_indices:
                res_50, res_10 = all_vertical_resolutions[i]
                if res_50 is not None and res_10 is not None:
                    label = f"{sequence_names[i]} (MTF$_{{50\\%}}$: {res_50:.3f} cycles/mm, MTF$_{{10\\%}}$: {res_10:.3f} cycles/mm)"
                else:
                    label = f"{sequence_names[i]} (MTF$_{{50\\%}}$: N/A, MTF$_{{10\\%}}$: N/A)"
                plt.plot(all_vertical_freq[i], all_vertical_mtf[i], label=label)

                plt.hlines(0.1,min(all_vertical_freq[i]),res_10,linestyles="--", color="black")
                plt.vlines(res_10,0,0.1,linestyles="--", color="black")
                plt.text((res_10-min(all_vertical_freq[i]))*(2/5),0.105,"10")

                plt.hlines(0.5,min(all_vertical_freq[i]),res_50,linestyles="--", color="black")
                plt.vlines(res_50,0,0.5,linestyles="--", color="black")
                plt.text((res_50-min(all_vertical_freq[i]))*(2/5),0.505,"50")
                
            plt.title(r'Phase Modulation Transfer Function')
            plt.xlabel('Spatial Frequency (cycles/mm)')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            

        else:
            print("No ROI selected")
            plt.close()

    
###Exemple d'utilisation###
dicom_directory = "DOCTORAT/10001_t2_tse_tra_p3_256_Bmed" # Path vers le dossier contenant les images DICOM
slice_index = 5 # Numéros de l'image 
save_to_csv = True 
Image = DicomImage(dicom_directory, slice_index, save_to_csv=True)



#Les trois méthodes de la classe

Image.plot_MTF()
# Image.compute_SNR_pixel()
# Image.plot_SNR_frequency()
