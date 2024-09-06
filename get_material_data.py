from mp_api.client import MPRester
import numpy as np
import matplotlib.pyplot as plt
import csv



with open('material_data.csv','w') as write_file:
    data_writer = csv.writer(write_file)
    
    with open('table_export.csv','r') as read_file:
        data_reader = csv.reader(read_file)
        material_ids = [] 
        for row in data_reader:
            material_ids.append(row[2])

        material_ids.pop(0)

        with MPRester(api_key="SFTntPIf0QnTef7f6sZKkBSwGILrxlXZ") as mpr:
            for material in material_ids:
                bandstructure = mpr.get_bandstructure_by_material_id(material)

                for element in bandstructure.bands:
                    for i in range(bandstructure.nb_bands):
                        band = list(bandstructure.bands.values())[0][i] - bandstructure.efermi
                        krange = np.linspace(-np.pi, np.pi, len(band))
                        poly_coeffs = np.polyfit(krange, band, 20)
                        data_writer.writerow(poly_coeffs)
