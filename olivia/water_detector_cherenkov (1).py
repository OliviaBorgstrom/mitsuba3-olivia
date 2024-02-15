# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:02:45 2023

@author: james
"""

"""
GET RID OF BACKGROUND LUMINANCE
CONVERT CANDELAS IN LUMINANCE
WORK OUT AVERAGE TIME TO TRAVEL FROM REFLECTION PLANE
=> GET ENERGY/PIXEL
FIND ENERGY/PHOTON BASED ON CANDELA STANDARD (555 nm)
=> GET PHOTONS/PIXEL


OPENEXR DEFINES THE LUMINANCE IN NITS (LUMENS/STERADIAN/M^2)
=> USE 2PI STERADIAN AS CAN ONLY EMIT OVER HALF PLANE
LUMENS ARE IN JOULES -> POWER/FREQ W/Hz
=> POWER IS LUMEN * FREQ
=> NO. PHOTONS = (POWER * TIME TO TRAVEL) / PHOTON_ENERGY
"""

import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import numpy as np
mi.set_variant("llvm_ad_rgb")

C_IN_WATER = (2.99792458*(10**8))/1.333 #in m/s
CANDELA_FREQ = 540*(10**12) #(in Hz)
CANDELA_PHOTON_ENERGY = 6.62607015*(10**-34) * CANDELA_FREQ #in J
AREA_PER_PIXEL = 0.01**2 #(in m^2)

#scene = mi.load_file("D:/Users/james/Documents/MPhys_Projects/water_detector_cherenkov_modified.xml")
#scene = mi.load_file("D:/Users/james/Documents/mitsuba3-manchester/MPhys/Rich_Geometry/richGeo.xml")

#photon_positions = np.genfromtxt("photons_detected_full.csv", delimiter=",")
photon_positions = np.genfromtxt("initialCringData.csv", delimiter=",")
x_pos = photon_positions[:,0] * 0.001
y_pos = photon_positions[:,1] * 0.001
z_pos = photon_positions[:,2] * 0.001
x_tar = (x_pos + photon_positions[:,3])
y_tar = (y_pos + photon_positions[:,4])
z_tar = (z_pos + photon_positions[:,5])
no_photons = x_pos.size

photon_info_temp = np.array((x_pos, y_pos, z_pos, x_tar, y_tar, z_tar))

photon_info = [no_photons]
for i in range (0, no_photons):
    photon_info.append(x_pos[i])
    photon_info.append(y_pos[i])
    photon_info.append(z_pos[i])
    photon_info.append(x_tar[i])
    photon_info.append(y_tar[i])
    photon_info.append(z_tar[i])
    
result = np.zeros((1, 1, len(photon_info)), dtype=np.float32)
result[0, 0, :] = photon_info
photon_list = mi.VolumeGrid(result)

distances_to_centre = np.zeros((100,100))
for i in range(0,100):
    y_distance = -0.495 + 0.01*i
    for j in range(0,100):
        x_distance = -0.495 + 0.01*j
        distance = (x_distance**2 + y_distance**2 + 0.49**2)**0.5
        distances_to_centre[j,i] = distance
        
times_to_centre = distances_to_centre/C_IN_WATER

print(photon_positions.size)
"""
spectrum = mi.load_dict({
    "type": "uniform",
    "value": 10000.0
    })

lum_photon_emitter = mi.load_dict({
    "type" : "photon_emitter",
    "photon_list" : photon_list,
    "id" : "lum_photon_emitter"
    })
"""

"""
"lum_area_light": {
    "type": "sphere",
    "to_world": mi.ScalarTransform4f.scale([1.15,1.15,1.15]).translate([0,0,3]),
    "emitter": {
        "type" : "area",
        "radiance" : {
            "type": "rgb",
            "value": [0.8, 1, 1]
            }
        }
    },
"lum_area_light_2": {
    "type": "sphere",
    "to_world": mi.ScalarTransform4f.scale([1.15,1.15,1.15]).translate([0,0,-3]),
    "emitter": {
        "type" : "area",
        "radiance" : {
            "type": "rgb",
            "value": [1, 0.6, 0.8]
            }
        }
    },
"""

scene = mi.load_dict({
    "type" : "scene",
    "scene_integrator" : {
        "type": "ptracer",
        "max_depth": 20,
        "hide_emitters": False
        },
    #FIX CROP SIZE
    "scene_sensor": {
        "type" : "perspective",
        "fov": 90,
        "to_world": mi.ScalarTransform4f.look_at(origin=[0, 0, 0],
                                                 target=[0, 0, 10],
                                                 up=[0,1,0]),
        "scene_film": {
            "type" : "hdrfilm",
            "pixel_format": "rgb",
            "width" : 100,
            "height" : 100,
            "file_format": "openexr",
            "rfilter": {
                "type": "box",
                #"stddev": 0.25
                }
            }
        },
    "scene_sampler": {
        "type": "independent",
        "sample_count": 16
        },

#insert lights here

    #"lum_area_light": {
     #   "type": "sphere",
      #  "to_world": mi.ScalarTransform4f.scale([0.5,0.5,0.5]).translate([0,0,6]),
       # "emitter": {
        #    "type" : "area",
         #   "radiance" : {
          #      "type": "rgb",
           #     "value": [0.8*5, 1*5, 1*5]
            #    }
            #}
        #},
    
    #"lum_area_light_2": {
     #   "type": "sphere",
      #  "to_world": mi.ScalarTransform4f.scale([0.5,0.5,0.5]).translate([0,0,-3]),
       # "emitter": {
        #    "type" : "area",
         #   "radiance" : {
          #      "type": "rgb",
           #     "value": [1, 0.6, 0.8]
            #    }
            #}
       # },
    
    "lum_env": {
        "type": "envmap",
        "filename": "neurathen_rock_castle_2k.exr"
        },


#    "sh_water_air_medium": {
 #       "type": "cube",
  #      "to_world": mi.ScalarTransform4f.scale([1,1,1.5]),
   #     "mat_water": {
    #        "type": "dielectric",
     #       "int_ior": "water",
      #      "ext_ior": "air"
       #     }
        #},
    #"sh_detection_plane": {
        #"type": "rectangle",
        #"to_world": mi.ScalarTransform4f.scale([0.5,0.5,0.5]).translate([0,0,0.49]),
        #"mat_detector_reflect" : {
            #"type": "twosided",
            #"material": {
                #"type" : "roughconductor",
                #"alpha" : 0.001,
                #"material": "Au",
                #"reflectance": {
                    #"type": "rgb",
                    #"value": [1.0,1.0,1.0]
                    #}
                #}
            #}
        #},
    "sh_spherical_mirror": {
        "type": "obj",
        "filename": "sh_spherical_mirror_alt.obj",
        "to_world": mi.ScalarTransform4f.rotate([1,0,0], 90).translate([0,0,0]),
        "mat_detector_reflect": {
            "type": "twosided",
            "material": {
                "type": "roughconductor",
                "alpha": 0.0,
                }
            }
        },
    
    "lum_photon_emitter": {
        "type": "photon_emitter",
        "photon_list": photon_list,
        }
    })

#lum_photon_emitter['intensity'] = 10000

params = mi.traverse(scene)
print(mi.traverse(scene))
#print(params['HDRFilm.size'])
#print(params['HDRFilm.crop_size'])
#print(params['scene_sensor.film.size'])
#print(params['scene_sensor.film.crop_size'])

#params['scene_sensor.film.size'] = [1080,1080]
#params['scene_sensor.film.crop_size'] = [1080,1080]
#print(params['scene_sensor.film.size'])
#print(params['scene_sensor.film.crop_size'])
image = mi.render(scene, spp=8)
image_arr = np.array(image)

#convert to lumens
lumens = image_arr * (AREA_PER_PIXEL**2)/(0.49**2)
lumens = lumens[:, :, 0]

#convert to radiating power
#radiated_power = lumens * CANDELA_FREQ

#convert to photons => multiplication of like elements only
photons_detected = np.zeros((100,100))
for i in range(0,100):
    for j in range(0,100):
        photons_detected[i,j] = lumens[i,j]/683 * times_to_centre[i,j]/CANDELA_PHOTON_ENERGY

print(photons_detected)
#image_arr -= image_arr.min()
#image_arr /= image_arr.max()

#print(params['glass.eta'])
#print(params['mirror.eta.value'])
#print(params['mirror.k.value'])
#print(params['mirror.specular_reflectance.value'])

print("Render finished!")
plt.axis("off")
plt.imshow(image ** (1/2.2))
plt.savefig("output.png")
#plt.close()

"""
fig, (ax, ax2) = plt.subplots(1,2)
ax.set_xlim([-495,495])
ax2.set_ylim([-495,495])
ax.autoscale(False)
ax2.autoscale(False)
ax.set_box_aspect(1)
ax2.set_box_aspect(1)

ax.imshow(image_arr, extent=[-495, 495, -495, 495])
plt.colorbar(mappable=ax.imshow(image_arr, extent=[-495, 495, -495, 495]), ax=ax)
ax.set_xlabel("X Position (mm)")
ax.set_ylabel("Y Position (mm)")
ax.set_title("Detected Luminance (nits)")

ax2.imshow(photons_detected, extent=[-495, 495, -495, 495])
plt.colorbar(mappable=ax2.imshow(photons_detected, extent=[-495, 495, -495, 495]), ax=ax2)
ax2.set_xlabel("X Position (mm)")
ax2.set_ylabel("Y Position (mm)")
ax2.set_title("Detected Photons")

plt.suptitle("Gaussian Filter $\sigma =$ 0.25, 128 Samples")
plt.tight_layout()
#ax.imshow(image ** (1/2.2))
plt.savefig("D:/Users/james/Documents/MPhys_Projects/Renders/cring_full_gaus128_twin.png")
plt.show()
#mi.util.write_bitmap("D:/Users/james/Documents/MPhys_Projects/Renders/photon_emitter_central_box_100px.png", image)
"""