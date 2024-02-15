import mitsuba as mi
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

### define functions ####
def ev_to_nm (energy):
    """
    Convert photon energy in eV to wavelength in nm
    """
    return (float(6.6260715e-34)*float(3.00e8)*float(1e9))/(float(1.6021e-19)*float(1e6))/energy

def quantum_efficiency(row):
    """
    Estimate quantum efficiency for a given wavelength
    Returns True if photon is accepted, False otherwise
    """
    # binned quantum efficiency values
    wavelength_steps    = [100, 200, 230,  270,  300,  330,  370,  400,  430,  470,  500,  530,  570,  600,  630,  670,   700, 1000]
    qe_steps            = [0,   0,   0.02, 0.20, 0.31, 0.35, 0.35, 0.33, 0.31, 0.24, 0.18, 0.08, 0.05, 0.02, 0.01, 0.002, 0,   0]
    # find index of the nearest wavelength step: lambda[idx - 1] < value <= lambda[idx]
    idx = np.searchsorted(wavelength_steps, row["Wavelength (nm)"])
    # return the corresponding quantum efficiency by linera interpolation
    a = (qe_steps[idx] - qe_steps[idx - 1])/(wavelength_steps[idx] - wavelength_steps[idx - 1])
    b = qe_steps[idx-1] - a * wavelength_steps[idx-1]
    qe_estimate = a*row["Wavelength (nm)"] + b
    # return True is photon is accepted, False otherwise
    return qe_estimate > random.uniform(0, 1)

def intersection_line_plane_v3(point_1, point_2, plane_coord, plane_normal, epsilon=1e-6):
    """
    Calculate the intersection between a line and a plane
    point_1, point_2: Define the line (arrays).
    plane_coord, plane_normal: define the plane:
        plane_coord Is a point on the plane (array)
        plane_normal Is a normal vector defining the plane direction (array);

    Return a Vector or None (when the intersection can't be found).
    """
    u = np.array(point_2) - np.array(point_1)
    dot = np.dot(np.array(plane_normal), np.array(u))
    if abs(dot) > epsilon:
        # The factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # Otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = np.array(point_1) - np.array(plane_coord)
        factor = - np.dot(np.array(plane_normal), np.array(w)) / dot
        u = factor * u
        return point_1 + u
    # The segment is parallel to plane.
    return None

def find_intersection(photon_data):
    intersection = []
    mirror_coord = [0, 2000, 355]
    mirror_normal = [0, 1000, 600]
    for i in range(len(photon_data)):
        photon_position = [photon_data.loc[i, "x"], photon_data.loc[i, "z"], photon_data.loc[i, "y"]]
        photon_target = [photon_data.loc[i, "x"] + photon_data.loc[i, "px"], photon_data.loc[i, "z"] + photon_data.loc[i, "pz"], photon_data.loc[i, "y"] + photon_data.loc[i, "py"]]
        intersection.append(intersection_line_plane_v3(photon_position, photon_target, mirror_coord, mirror_normal))
    return intersection

def acceptance_criteria(row, intersection):
    """
    Criteria based on the geometrical acceptance of the detector
    Takes as input an array of intersection points for each photons
    Returns True if photon is within detector, False otherwise
    """
    if abs(intersection[row.name][0]) > 750 or not (30 <= intersection[row.name][2] <= 680):
        return False
    return True

def reflectivity_criteria(row, intersection):
    """
    Criteria based on the reflectivity of the mirrors
    Return True if photon is reflected, False otherwise
    """
    # bin reflectivity of the four spherical mirrors
    wavelength_steps = np.array([200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600])
    reflectivity_1  = [0.91, 0.875, 0.86, 0.875, 0.9, 0.915, 0.92, 0.925, 0.925, 0.92, 0.915, 0.905, 0.895, 0.885, 0.875, 0.865]
    reflectivity_2  = [0.905, 0.87, 0.86, 0.875, 0.90, 0.915, 0.92, 0.925, 0.925, 0.92, 0.915, 0.905, 0.895, 0.885, 0.88, 0.865]
    # reflectivity_3  = [0.885, 0.865, 0.89, 0.91, 0.92, 0.925, 0.925, 0.925, 0.92, 0.915, 0.905, 0.89, 0.885, 0.87, 0.86, 0.855]
    # reflectivity_4  = [0.90, 0.865, 0.86, 0.89, 0.905, 0.915, 0.925, 0.925, 0.92, 0.915, 0.91, 0.90, 0.89, 0.88, 0.87, 0.86]
    idx_lower = np.searchsorted(wavelength_steps, row["Wavelength (nm)"], side='right') - 1
    # accept the photons when outside the wavelength range (assume 100% reflectivity)
    if row["Wavelength (nm)"] < wavelength_steps[0] or row["Wavelength (nm)"] > wavelength_steps[-1]:
        return True
    # if the photon hits the left mirror use the first reflectivity data
    if intersection[row.name][0] < 0:
        return random.uniform(0, 1) <= reflectivity_1[idx_lower]
    # if the photon hits the right mirror use the second reflectivity data
    return random.uniform(0, 1) <= reflectivity_2[idx_lower]

def generate_emitter_data(photon_data):
    """
    Generates the data for the photon_emitter plugin of Mitsuba using the photon data
    """
    x_position, y_position, z_position = photon_data.values[:, 1:4].T
    x_momentum, y_momentum, z_momentum = photon_data.values[:, 4:7].T
    # calculate the target coordinates of the photons
    x_target = x_position + x_momentum
    y_target = y_position + y_momentum
    z_target = z_position + z_momentum
    # combine them into a single array
    emitter_data = np.column_stack((x_position, y_position, z_position, x_target, y_target, z_target)).flatten()
    emitter_data = np.insert(emitter_data, 0, len(x_position))
    # create a 3D array of the emitter data
    result = np.zeros((1, 1, len(emitter_data)), dtype=np.float32)
    result[0, 0, :] = emitter_data
    return result

### main ###
# load data from original G4 output (csv format)
column_names = ["time (ps)", "x", "y", "z", "px", "py", "pz", "E (MeV)"]
photon_data_full = pd.read_csv('photons_with_wavelength.csv', names = column_names)
# convert energy column to wavelength in nm and rename it
photon_data_full['E (MeV)'] = photon_data_full['E (MeV)'].apply(ev_to_nm)
photon_data_full.rename(columns={"E (MeV)": "Wavelength (nm)"}, inplace = True)
# apply the quantum efficiency of the detector
photon_detected = photon_data_full[photon_data_full.apply(quantum_efficiency, axis=1)]
# reset the indices of the dataframe
photon_detected.reset_index(drop=True, inplace= True)
# Calculate the intersection of the photon with the mirror plane
intersection = find_intersection(photon_detected)
# apply the acceptance criteria
photon_detected = photon_detected[photon_detected.apply(acceptance_criteria, axis=1, intersection=intersection)]
photon_detected.reset_index(drop=True, inplace= True)
# apply reflectivity criteria
photon_detected = photon_detected[photon_detected.apply(reflectivity_criteria, axis=1, intersection=intersection)]
photon_detected.reset_index(drop=True, inplace= True)
# account for the reflectivity of the plane mirror by rejecting 10% of the photons
photon_detected = photon_detected.sample(frac=0.9, ignore_index=True)
photon_detected.reset_index(drop=True, inplace= True)
# swap the x and y coordinates of the photons in position and momentum
photon_detected[["x", "y", "px", "py"]] = photon_detected[["y", "x", "py", "px"]].values
# set the mitsuba variant
mi.set_variant("llvm_ad_rgb")
# create the emitter data
photon_list = mi.VolumeGrid(generate_emitter_data(photon_detected))
# create the scene
scene_description = {
    'type': 'scene',
    
    'integrator': { 
        'type': 'ptracer', 
        'max_depth': 65,
        'hide_emitters': False,
    },

    'sensor': { 
        'type': 'perspective',
        'fov': 100,
        'to_world': mi.ScalarTransform4f.look_at(origin=[0, 0, 0],
                                                 target=[0, 0, 10],
                                                 up=[0,1,0]),
        'sampler': {
            'type': 'independent',
            'sample_count': 32,
        },
        'film': {
            'type': 'hdrfilm',
            'width': 1024,
            'height': 1024,
            'file_format': 'openexr',
            'pixel_format': 'rgb',
            'component_format': 'uint32',
            'filter' : {
                'type': 'tent',
            },
        },
    },

    'MirrorBSDF' : {
        'type' : 'conductor',
        'material' : 'none',
    },

    'mirrors' : {
        'type' : 'obj',
        'filename' : 'mirrors_version4.obj',
        # 'to_world': mi.ScalarTransform4f.look_at(origin=[200, 500, 0],
        #                                          target=[200, 2000, 0],
        #                                          up=[0, 0, 1]),
        'bsdf_id' : {
            'type' : 'ref',
            'id' : 'MirrorBSDF',
        },
    },

    'test' : {
        'type' : 'diffuse',
        'reflectance': {
            'type' : 'rgb',
            'value': [1, 1, 1],
        },
    },

    'structure' : {
        'type' : 'obj',
        'filename' : 'RichGeo1.obj',
        # 'to_world': mi.ScalarTransform4f.look_at(origin=[200, 500, -0.5],
        #                                          target=[200, 1600, -0.5],
        #                                          up=[0, 0, 1]),
        'bsdf_id' : {
            'type' : 'ref',
            'id' : 'test',
        },
    },

    'photons' : {
        'type' : 'photon_emitter',
        'photon_list' : photon_list,
    },
}
# load the scene
scene = mi.load_dict(scene_description)
# render the scene into a bitmap
image = mi.render(scene)
# save the image in a png file
mi.util.write_bitmap("sphere_spectral.png",image)
# convert bitmap to numpy array
bmp = mi.Bitmap(image)
# convert bitmap to numpy array
bmp_np = np.array(bmp); print(bmp_np.flatten())

if not bmp_np.any():
    print("nothing here")
else:
    print("something found")
