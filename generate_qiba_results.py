from ADC_MRI_extractor import ADC_MRI_extractor
from DicomImage_QIBA import DicomImage

slice_index = 17
path = "/home/lbsc/IRM_diffusion/15_nov_2025_ANT_TETE/DRB"
extractor_obj = ADC_MRI_extractor(path=path)
Image = DicomImage(dicom_directory=path, slice_index=slice_index)






extractor_obj.data_extractor()
Image.b_value_split()
# Image.compute_MTF()
Image.create_ADC_maps_least_squares()
Image.compute_qiba_metrics_13_reg(shape_ROI="Circular")


# Obsolete
# Image.create_ADC_maps()
