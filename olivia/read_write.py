import numpy as np
import xml.etree.ElementTree as ET
import mitsuba as mi
import matplotlib.pyplot as plt
import pandas as pd
import random

nm_per_ev_constant = (float(6.6260715e-34)*float(3.00e8)*float(1e9))/(float(1.6021e-19)*float(1e6))

#Values for linear interpolation.
wavelength_steps = [100, 200, 230, 270, 300, 330, 370, 400, 430, 470, 500, 530, 570, 600, 630, 670, 700, 1000]
qe_steps = [0, 0, 0.02, 0.20, 0.31, 0.35, 0.35, 0.33, 0.31, 0.24, 0.18, 0.08, 0.05, 0.02, 0.01, 0.002, 0, 0]
rows_to_drop =[]

#load data from original G4 output (csv format)
column_names = ["time (ps)", "x", "y", "z", "px", "py", "pz", "E (MeV)"]
photon_data_full = pd.read_csv('photons_with_wavelength.csv', names = column_names)
initial_number_of_photons = photon_data_full.count()[0]

def ev_to_nm (energy):
    return nm_per_ev_constant/energy

#gives linear relation between two wavelegths thanks to two efficiency values
def linear_interp (j,w_input):
    a = (qe_steps[j] - qe_steps[j-1])/(wavelength_steps[j] - wavelength_steps[j-1])
    b = qe_steps[j-1] - a*wavelength_steps[j-1]
    return a*w_input+b

#Convert energy column to wavelength in nm.
photon_data_full['E (MeV)'] = photon_data_full['E (MeV)'].apply(ev_to_nm)
photon_data_full.rename(columns={"E (MeV)": "Wavelength (nm)"}, inplace = True)

#iterate over each photon.
for i in range(initial_number_of_photons):
    for j in range(len(wavelength_steps)):
        #find index in wavelength_steps corresponding to the photon.
        if photon_data_full.loc[i, 'Wavelength (nm)']<wavelength_steps[j]:
            break
    #Use linear interpolation to calculate QE at this wavelength.
    qe_estimated = linear_interp(j, photon_data_full.loc[i, 'Wavelength (nm)']) 
    x = random.uniform(0, 1)
    if x > qe_estimated:
        rows_to_drop.append(i)

#generate new frame only with photons that will be detected.
photon_detected = photon_data_full.drop(rows_to_drop)
#I don't know why a first column is created and keeps track of the old index of photons.
photon_detected.reset_index(drop=True, inplace= True)

#Can be removed later. Outputs characteristics of saved photons.
photon_detected.to_csv('./photons_detected_spectral.csv' )
final_number_of_photons = photon_detected.count()[0]
fraction_detected = final_number_of_photons/initial_number_of_photons
print ("{0:.4} of emitted photon in G4 are actually detected.".format(fraction_detected))

x_position = photon_detected.values[:,1] # x position of the detector
y_position = photon_detected.values[:,2] # y position of the detectore
z_position = photon_detected.values[:,3] # z position of the detector
x_momentum = photon_detected.values[:,4] # x momentum of the particle
y_momentum = photon_detected.values[:,5] # y momentum of the particle
z_momentum = photon_detected.values[:,6] # z momentum of the particle

# calculate the target coordinates for each photons
x_target = []
y_target = []
z_target = []
for i in range(len(x_position)):
    x_target.append(x_position[i] + x_momentum[i])
    y_target.append(y_position[i] + y_momentum[i])
    z_target.append(z_position[i] + z_momentum[i])
    
tree = ET.parse('base_geometry.xml')
root = tree.getroot()
for emitter in root.iter('emitter'):
    print(emitter.attrib)
    
tree = ET.parse('base_geometry.xml')
root = tree.getroot()
for shape in root.iter('shape'):
    if shape.get('id') == 'mirror_spherical':
        for lookat in shape.iter('lookat'):
            # get mirror's position
            x_mirror = float(lookat.get('origin').split(',')[0])
            y_mirror = float(lookat.get('origin').split(',')[1])
            z_mirror = float(lookat.get('origin').split(',')[2])
            # get mirror's target i.e. the vector normal to the mirror plane
            x_target_mirror = float(lookat.get('target').split(',')[0])
            y_target_mirror = float(lookat.get('target').split(',')[1])
            z_target_mirror = float(lookat.get('target').split(',')[2])

print(x_mirror, y_mirror, z_mirror)
print(x_target_mirror, y_target_mirror, z_target_mirror)

# calculate the intersection between of line and a plane
def isect_line_plane_v3(point_1, point_2, plane_coord, plane_normal, epsilon=1e-6):
    """
    point_1, point_2: Define the line.
    plane_coord, plane_normal: define the plane:
        plane_coord Is a point on the plane (plane coordinate).
        plane_normal Is a normal vector defining the plane direction;

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

intersection = []
distance = []
mirror_coord = [x_mirror, y_mirror, z_mirror]
mirror_normal = [x_target_mirror, y_target_mirror, z_target_mirror]
for i in range(len(x_position)):
    point_1 = [x_position[i], y_position[i], z_position[i]]
    point_2 = [x_target[i], y_target[i], z_target[i]]
    intersection.append(isect_line_plane_v3(point_1, point_2, mirror_coord, mirror_normal, epsilon=1e-6))
    distance.append(np.sqrt((x_position[i] - intersection[i][0])**2 + (y_position[i] - intersection[i][1])**2 + (z_position[i] - intersection[i][2])**2))
print(distance)

# # calculate the distance between the photons and the mirror
# distance = []
# for i in range(len(x_position)):
#     distance.append(np.sqrt((x_position[i] - x_mirror)**2 + (y_position[i] - y_mirror)**2 + (z_position[i] - z_mirror)**2))
# calculate the cutoff angle for each spot light
cutoff_angle = np.array(2 * np.arctan(100/np.array(distance))) # should give a cutoff angle of around 0.2 degrees
print(len(cutoff_angle))

def add_spot_emitter(x, z, y, x_target, z_target, y_target, cutoff_angle):
    
    # parsing the xml file of the scene
    tree = ET.parse('base_geometry.xml')
    root = tree.getroot()
    # add a spot emitter to the scene
    ### here, the y and z axis are switched (because the scene is rotated by 90 degrees?) ###
    for i in range(len(x)):
        new_emitter = ET.SubElement(root, 'emitter', attrib={'type':'photon_emitter_olivia'})
        transform = ET.SubElement(new_emitter, 'transform', attrib={'name':'to_world'})
        lookat = ET.SubElement(transform, 'lookat', attrib={'origin':str(x[i]) + ', ' + str(z[i]) + ', ' + str(y[i]), 'target':str(x_target[i]) + ', ' + str(z_target[i]) + ', ' + str(y_target[i]), 'up':'0,1,0'})
        rgb = ET.SubElement(new_emitter, 'rgb', attrib={'name':'intensity', 'value':'200000000.0'})
        # modify the cutoff angle of the light
        float = ET.SubElement(new_emitter, 'float', attrib={'name':'cutoff_angle', 'value':str(cutoff_angle[i])})
    
    # write it all on a new xml file.
    tree.write("full_geometry.xml")

# add spot emitters
add_spot_emitter(x_position, z_position, y_position, x_target, z_target, y_target, cutoff_angle)

# printing the attributes of all emitters
# tree = ET.parse('real_geometry.xml')
# root = tree.getroot()
# for emitter in root.iter('emitter'):
#     for lookat in emitter.iter('lookat'):
#         print(lookat.attrib)

# test cutoff values of the emitter
tree = ET.parse('full_geometry.xml')
root = tree.getroot()
for emitter in root.iter('emitter'):
    for float in emitter.iter('float'):
        print(float.get('value'))
        
mi.set_variant('scalar_rgb')
scene = mi.load_file('full_geometry.xml')
image = mi.render(scene)
plt.figure(figsize = (20,20))
plt.axis('off')
plt.imsave('test.png',image ** (1.0 / 2.2)); 