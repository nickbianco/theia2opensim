import os
import opensim as osim
import numpy as np
from c3d import C3D
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)

def compute_scale_factor(state, frame1, frame2, pos1, pos2):
    # Magnitude of relative position between the two model frames.
    frame1_pos = frame1.getPositionInGround(state).to_numpy()
    frame2_pos = frame2.getPositionInGround(state).to_numpy()
    frame_rel_mag = np.linalg.norm(frame2_pos - frame1_pos)

    # Magnitude of relative position between the two Theia frames.
    pos_rel_mag = np.linalg.norm(pos2.to_numpy() - pos1.to_numpy())

    scale_factor = pos_rel_mag / frame_rel_mag
    return scale_factor

# Data import
# -----------
# Load the model generic model.
model = osim.Model('unscaled_generic.osim')
state = model.initSystem()

# Adjust hip joint offset
hip_joint_offset = 0.03  # 2 cm
hip_r_pelvis_offset = osim.PhysicalOffsetFrame.safeDownCast(
    model.updComponent('/jointset/hip_r/pelvis_offset'))
hip_r_pelvis_offset.upd_translation()[2] += hip_joint_offset
hip_l_pelvis_offset = osim.PhysicalOffsetFrame.safeDownCast(
    model.updComponent('/jointset/hip_l/pelvis_offset'))
hip_l_pelvis_offset.upd_translation()[2] -= hip_joint_offset
model.finalizeConnections()
model.initSystem()

# Import the C3D file.
data_path = os.path.join(config['data_path'], 'acl', 'jump_1')
c3d_fpath = os.path.join(data_path, 'pose_0.c3d')
c3d = C3D(c3d_fpath)

# Extract relevant frame positions from position data.
positions = c3d.get_frame_positions_table()
trc = osim.TRCFileAdapter()
trc.write(positions, 'positions.trc')

pelvis_pos = positions.getDependentColumn('pelvis')[0]
torso_pos = positions.getDependentColumn('torso')[0]
l_uarm_pos = positions.getDependentColumn('l_uarm')[0]
r_uarm_pos = positions.getDependentColumn('r_uarm')[0]
l_larm_pos = positions.getDependentColumn('l_larm')[0]
r_larm_pos = positions.getDependentColumn('r_larm')[0]
r_hand_pos = positions.getDependentColumn('r_hand')[0]
l_hand_pos = positions.getDependentColumn('l_hand')[0]
l_thigh_pos = positions.getDependentColumn('l_thigh')[0]
r_thigh_pos = positions.getDependentColumn('r_thigh')[0]
l_shank_pos = positions.getDependentColumn('l_shank')[0]
r_shank_pos = positions.getDependentColumn('r_shank')[0]
l_foot_pos = positions.getDependentColumn('l_foot')[0]
r_foot_pos = positions.getDependentColumn('r_foot')[0]
l_toes_pos = positions.getDependentColumn('l_toes')[0]
r_toes_pos = positions.getDependentColumn('r_toes')[0]

# Extract corresponding model frames.
lower_pelvis = osim.PhysicalFrame.safeDownCast(
    model.getComponent('/bodyset/pelvis/lower_pelvis'))
upper_torso = osim.PhysicalFrame.safeDownCast(
    model.getComponent('/bodyset/torso/upper_torso'))
l_uarm = osim.PhysicalFrame.safeDownCast(
    model.getComponent('/jointset/acromial_l/humerus_l_offset/l_uarm'))
r_uarm = osim.PhysicalFrame.safeDownCast(
    model.getComponent('/jointset/acromial_r/humerus_r_offset/r_uarm'))
l_larm = osim.PhysicalFrame.safeDownCast(
    model.getComponent('/jointset/elbow_l/ulna_l_offset/l_larm'))
r_larm = osim.PhysicalFrame.safeDownCast(
    model.getComponent('/jointset/elbow_r/ulna_r_offset/r_larm'))
r_hand = osim.PhysicalFrame.safeDownCast(
    model.getComponent('/jointset/radius_hand_r/hand_r_offset/r_hand'))
l_hand = osim.PhysicalFrame.safeDownCast(
    model.getComponent('/jointset/radius_hand_l/hand_l_offset/l_hand'))
l_thigh = osim.PhysicalFrame.safeDownCast(
    model.getComponent('/jointset/hip_l/femur_l_offset/l_thigh'))
r_thigh = osim.PhysicalFrame.safeDownCast(
    model.getComponent('/jointset/hip_r/femur_r_offset/r_thigh'))
l_shank = osim.PhysicalFrame.safeDownCast(
    model.getComponent('/jointset/walker_knee_l/tibia_l_offset/l_shank'))
r_shank = osim.PhysicalFrame.safeDownCast(
    model.getComponent('/jointset/walker_knee_r/tibia_r_offset/r_shank'))
l_foot = osim.PhysicalFrame.safeDownCast(
    model.getComponent('/jointset/ankle_l/talus_l_offset/l_foot'))
r_foot = osim.PhysicalFrame.safeDownCast(
    model.getComponent('/jointset/ankle_r/talus_r_offset/r_foot'))
l_toes = osim.PhysicalFrame.safeDownCast(
    model.getComponent('/jointset/mtp_l/toes_l_offset/l_toes'))
r_toes = osim.PhysicalFrame.safeDownCast(
    model.getComponent('/jointset/mtp_r/toes_r_offset/r_toes'))

# Create scale factors
# --------------------
scaleset = osim.ScaleSet()

# Pelvis
pelvis_scale = osim.Scale()
pelvis_scale.setSegmentName('pelvis')
pelvis_factors = osim.Vec3(1.0)
pelvis_factors[1] = compute_scale_factor(
        state, lower_pelvis, upper_torso, pelvis_pos, torso_pos)
pelvis_factors[2] = compute_scale_factor(
        state, l_thigh, r_thigh, l_thigh_pos, r_thigh_pos)
pelvis_scale.setScaleFactors(pelvis_factors)
scaleset.cloneAndAppend(pelvis_scale)

# Torso
torso_scale = osim.Scale()
torso_scale.setSegmentName('torso')
torso_factors = osim.Vec3(1.0)
torso_factors[1] = compute_scale_factor(
    state, lower_pelvis, upper_torso, pelvis_pos, torso_pos)
torso_factors[2] = compute_scale_factor(
    state, l_uarm, r_uarm, l_uarm_pos, r_uarm_pos)
torso_scale.setScaleFactors(torso_factors)
scaleset.cloneAndAppend(torso_scale)

# Right humerus
humerus_r_scale = osim.Scale()
humerus_r_scale.setSegmentName('humerus_r')
humerus_r_factors = osim.Vec3(1.0)
humerus_r_factors[1] = compute_scale_factor(
    state, r_uarm, r_larm, r_uarm_pos, r_larm_pos)
humerus_r_scale.setScaleFactors(humerus_r_factors)
scaleset.cloneAndAppend(humerus_r_scale)

# Left humerus
humerus_l_scale = osim.Scale()
humerus_l_scale.setSegmentName('humerus_l')
humerus_l_factors = osim.Vec3(1.0)
humerus_l_factors[1] = compute_scale_factor(
    state, l_uarm, l_larm, l_uarm_pos, l_larm_pos)
humerus_l_scale.setScaleFactors(humerus_l_factors)
scaleset.cloneAndAppend(humerus_l_scale)

# Right radius
radius_r_scale = osim.Scale()
radius_r_scale.setSegmentName('radius_r')
radius_r_factors = osim.Vec3(1.0)
radius_r_factors[1] = compute_scale_factor(
    state, r_larm, r_hand, r_larm_pos, r_hand_pos)
radius_r_scale.setScaleFactors(radius_r_factors)
scaleset.cloneAndAppend(radius_r_scale)

# Right ulna
ulna_r_scale = osim.Scale()
ulna_r_scale.setSegmentName('ulna_r')
ulna_r_scale.setScaleFactors(radius_r_factors)
scaleset.cloneAndAppend(ulna_r_scale)

# Right hand
hand_r_scale = osim.Scale()
hand_r_scale.setSegmentName('hand_r')
hand_r_scale.setScaleFactors(radius_r_factors)
scaleset.cloneAndAppend(hand_r_scale)

# Left radius
radius_l_scale = osim.Scale()
radius_l_scale.setSegmentName('radius_l')
radius_l_factors = osim.Vec3(1.0)
radius_l_factors[1] = compute_scale_factor(
    state, l_larm, l_hand, l_larm_pos, l_hand_pos)
radius_l_scale.setScaleFactors(radius_l_factors)
scaleset.cloneAndAppend(radius_l_scale)

# Left ulna
ulna_l_scale = osim.Scale()
ulna_l_scale.setSegmentName('ulna_l')
ulna_l_scale.setScaleFactors(radius_l_factors)
scaleset.cloneAndAppend(ulna_l_scale)

# Left hand
hand_l_scale = osim.Scale()
hand_l_scale.setSegmentName('hand_l')
hand_l_scale.setScaleFactors(radius_l_factors)
scaleset.cloneAndAppend(hand_l_scale)

# Right femur
femur_r_scale = osim.Scale()
femur_r_scale.setSegmentName('femur_r')
femur_r_factors = osim.Vec3(1.0)
femur_r_factors[1] = compute_scale_factor(
    state, r_thigh, r_shank, r_thigh_pos, r_shank_pos)
femur_r_scale.setScaleFactors(femur_r_factors)
scaleset.cloneAndAppend(femur_r_scale)

# Left femur
femur_l_scale = osim.Scale()
femur_l_scale.setSegmentName('femur_l')
femur_l_factors = osim.Vec3(1.0)
femur_l_factors[1] = compute_scale_factor(
    state, l_thigh, l_shank, l_thigh_pos, l_shank_pos)
femur_l_scale.setScaleFactors(femur_l_factors)
scaleset.cloneAndAppend(femur_l_scale)

# Right tibia
tibia_r_scale = osim.Scale()
tibia_r_scale.setSegmentName('tibia_r')
tibia_r_factors = osim.Vec3(1.0)
tibia_r_factors[1] = compute_scale_factor(
    state, r_shank, r_foot, r_shank_pos, r_foot_pos)
tibia_r_scale.setScaleFactors(tibia_r_factors)
scaleset.cloneAndAppend(tibia_r_scale)

# Left tibia
tibia_l_scale = osim.Scale()
tibia_l_scale.setSegmentName('tibia_l')
tibia_l_factors = osim.Vec3(1.0)
tibia_l_factors[1] = compute_scale_factor(
    state, l_shank, l_foot, l_shank_pos, l_foot_pos)
tibia_l_scale.setScaleFactors(tibia_l_factors)
scaleset.cloneAndAppend(tibia_l_scale)

# Right calcaneus
calcaneus_r_scale = osim.Scale()
calcaneus_r_scale.setSegmentName('calcn_r')
calcaneus_r_factors = osim.Vec3(1.0)
calcaneus_r_factors[0] = compute_scale_factor(
    state, r_foot, r_toes, r_foot_pos, r_toes_pos)
calcaneus_r_factors[1] = compute_scale_factor(
    state, r_foot, r_toes, r_foot_pos, r_toes_pos)
calcaneus_r_scale.setScaleFactors(calcaneus_r_factors)
scaleset.cloneAndAppend(calcaneus_r_scale)

# Right talus
talus_r_scale = osim.Scale()
talus_r_scale.setSegmentName('talus_r')
talus_r_scale.setScaleFactors(calcaneus_r_factors)
scaleset.cloneAndAppend(talus_r_scale)

# Right toes
toes_r_scale = osim.Scale()
toes_r_scale.setSegmentName('toes_r')
toes_r_scale.setScaleFactors(calcaneus_r_factors)
scaleset.cloneAndAppend(toes_r_scale)

# Left calcaneus
calcaneus_l_scale = osim.Scale()
calcaneus_l_scale.setSegmentName('calcn_l')
calcaneus_l_factors = osim.Vec3(1.0)
calcaneus_l_factors[0] = compute_scale_factor(
    state, l_foot, l_toes, l_foot_pos, l_toes_pos)
calcaneus_l_factors[1] = compute_scale_factor(
    state, l_foot, l_toes, l_foot_pos, l_toes_pos)
calcaneus_l_scale.setScaleFactors(calcaneus_l_factors)
scaleset.cloneAndAppend(calcaneus_l_scale)

# Left talus
talus_l_scale = osim.Scale()
talus_l_scale.setSegmentName('talus_l')
talus_l_scale.setScaleFactors(calcaneus_l_factors)
scaleset.cloneAndAppend(talus_l_scale)

# Left toes
toes_l_scale = osim.Scale()
toes_l_scale.setSegmentName('toes_l')
toes_l_scale.setScaleFactors(calcaneus_l_factors)
scaleset.cloneAndAppend(toes_l_scale)

# Scale the model
# ---------------
model.scale(state, scaleset, True)

model.finalizeConnections()
state = model.initSystem()
model.setName('jump_1_scaled')
model.printToXML(os.path.join(data_path, 'jump_1_scaled.osim'))
