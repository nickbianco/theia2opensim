import os
import opensim as osim

model = osim.Model(os.path.join('models', 'RajagopalLaiUhlrich2023.osim'))
model.initSystem()



worldbody	pelvis	l_thigh	l_shank	l_foot	l_toes	r_thigh	r_shank	r_foot	r_toes	torso
l_clavicle	l_uarm	l_larm	l_hand	r_clavicle	r_uarm	r_larm	r_hand	head	pelvis_shifted
frame_map = {
    'pelvis': 'pelvis_offset',
    'l_thigh': 'femur_l_offset',
    'l_shank': 'tibia_l_offset',
    'l_foot': 'l_foot',
    'l_toes': 'l_toes',
    'r_thigh': 'r_thigh',
    'r_shank': 'r_shank',
    'r_foot': 'r_foot',
    'r_toes': 'r_toes',
    'torso': 'torso',
    'l_clavicle': 'l_clavicle',
    'l_uarm': 'l_uarm',
    'l_larm': 'l_larm',
    'l_hand': 'l_hand',
    'r_clavicle': 'r_clavicle',
    'r_uarm': 'r_uarm',
    'r_larm': 'r_larm',
    'r_hand': 'r_hand',
    'head': 'head',
    'pelvis_shifted': 'pelvis_shifted',
}