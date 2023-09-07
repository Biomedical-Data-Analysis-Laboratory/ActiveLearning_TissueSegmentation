import numpy as np

# map vips formats to np dtypes
format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

# map np dtypes to vips
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}

number_coded_uro_tissue = {
    'Grade 1 Low': 0,
    'Grade 2 Low': 1,
    'Grade 2 High': 2,
    'Grade 3 High': 3,
    'CIS': 4,
    'Dysplasia': 5,
    'Urothelium no subclass': 6,
    'Normal urothelium': 7,
    'Cancerous urothelium': 8,
    'Urothelium without immune cell infiltration': 9,
    'Urothelium With immune cell infiltration': 10,
    'Flat lesion': 11,
    'Variant histology': 12,
}

# Conversion factors between different magnification scale of SCN images
# To go UP in magnification level from (25x to 100x) or (25x to 400x) or (100x to 400x) use "divide"
# To go DOWN in magnification level from (100x to 25x) or (400x to 25x) or (400x to 100x) use "multiply"
Scale_between_100x_400x = 0.25
Scale_between_25x_100x = 0.25
Scale_between_25x_400x = 0.0625


def get_tissue_name_and_index_of_classes():
    name_and_index_of_classes = dict()
    name_and_index_of_classes[0] = {'display_name': 'Background', 'name': 'background', 'index': 0, 'used_in_training': 1}
    name_and_index_of_classes[1] = {'display_name': 'Blood', 'name': 'blood', 'index': 1, 'used_in_training': 1}
    name_and_index_of_classes[2] = {'display_name': 'Damaged', 'name': 'damaged', 'index': 2, 'used_in_training': 1}
    name_and_index_of_classes[3] = {'display_name': 'Muscle', 'name': 'muscle', 'index': 3, 'used_in_training': 1}
    name_and_index_of_classes[4] = {'display_name': 'Stroma', 'name': 'stroma', 'index': 4, 'used_in_training': 1}
    name_and_index_of_classes[5] = {'display_name': 'Urothelium', 'name': 'urothelium', 'index': 5, 'used_in_training': 1}
    name_and_index_of_classes[6] = {'display_name': 'Undefined', 'name': 'undefined', 'index': 6, 'used_in_training': 0}
    return name_and_index_of_classes


def get_tissue_name_and_index_of_classes_binary_mode():
    name_and_index_of_classes = dict()
    name_and_index_of_classes[0] = {'display_name': 'Urothelium', 'name': 'urothelium', 'index': 0, 'used_in_training': 1}
    name_and_index_of_classes[1] = {'display_name': 'Other', 'name': 'other', 'index': 1, 'used_in_training': 1}
    name_and_index_of_classes[2] = {'display_name': 'Undefined', 'name': 'undefined', 'index': 2, 'used_in_training': 0}
    return name_and_index_of_classes