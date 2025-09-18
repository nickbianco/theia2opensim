import os
import opensim as osim

# This maps Theia rotations to existing offset frames in the generic model. We will
# add a new frame to the model at the location of these offset frames, but with
# orientiation aligned with the ground frame (X forward, Y up, Z right).
frame_map = {
    'l_thigh': '/jointset/hip_l/femur_l_offset',
    'l_shank': '/jointset/walker_knee_l/tibia_l_offset',
    'l_foot': '/jointset/ankle_l/talus_l_offset',
    'l_toes': '/jointset/mtp_l/toes_l_offset',
    'r_thigh': '/jointset/hip_r/femur_r_offset',
    'r_shank': '/jointset/walker_knee_r/tibia_r_offset',
    'r_foot': '/jointset/ankle_r/talus_r_offset',
    'r_toes': '/jointset/mtp_r/toes_r_offset',
    'l_uarm': '/jointset/acromial_l/humerus_l_offset',
    'l_larm': '/jointset/elbow_l/ulna_l_offset',
    'l_hand': '/jointset/radius_hand_l/hand_l_offset',
    'r_uarm': '/jointset/acromial_r/humerus_r_offset',
    'r_larm': '/jointset/elbow_r/ulna_r_offset',
    'r_hand': '/jointset/radius_hand_r/hand_r_offset',
}

def add_theia_frame(model, state, theia_frame_name, model_frame_path, offset):
    # Get the generic model frame and its transform in the ground frame.
    frame = osim.PhysicalFrame.safeDownCast(model.updComponent(model_frame_path))
    T_GO = frame.getTransformInGround(state)

    # Create an identity rotation representing the orientation of the new frame with respect
    # to the ground frame, R_GF = I.
    R_GF = osim.Rotation()

    # The rotation for the new frame is the offset frame relative to the existing offset
    # frame in the model, or R_OF = R_OG * R_GF = (~R_GO) * R_GF.
    R_GO = T_GO.R()
    R_OG = osim.Rotation(R_GO.transpose())
    R_OF = R_OG.multiply(R_GF)

    # Create a new physical offset frame and add it the model as a subcomponent of the
    # existing.
    T_OF = osim.Transform(R_OF, offset)
    offset_frame = osim.PhysicalOffsetFrame(theia_frame_name, frame, T_OF)

    frame.addComponent(offset_frame)

# Load the generic model and weld the subtalar joints.
model_fpath = os.path.join('models', 'RajagopalLaiUhlrich2023.osim')
modelProcessor = osim.ModelProcessor(model_fpath)
jointsToWeld = osim.StdVectorString()
jointsToWeld.append('subtalar_l')
jointsToWeld.append('subtalar_r')
modelProcessor.append(osim.ModOpReplaceJointsWithWelds(jointsToWeld))
model = modelProcessor.process()

# The state represents the model in the default pose: the model is facing forward along
# the X axis, and the Z axis is pointing to the right (Y is up). The frame calculations
# below depend on the model being in this default pose. If the default pose changes, it
# will change how the frames are added to the model.
state = model.initSystem()

# Add new frames to the model based on the Theia data.
zero_offset = osim.Vec3(0)
for theia_frame_name in frame_map.keys():
    add_theia_frame(model, state, theia_frame_name, frame_map[theia_frame_name],
                    zero_offset)

# Add pelvis frame.
# ----------------
# This frame is the located at the midpoint between the left and right hip joint,
# expressed in the pelvis frame.
state = model.initSystem()
ground = model.getGround()
pelvis = osim.Body.safeDownCast(
        model.getComponent('/bodyset/pelvis'))
left_hip = osim.PhysicalOffsetFrame.safeDownCast(
        model.getComponent('/jointset/hip_l/femur_l_offset'))
right_hip = osim.PhysicalOffsetFrame.safeDownCast(
        model.getComponent('/jointset/hip_r/femur_r_offset'))

p_left = left_hip.getPositionInGround(state)
p_right = right_hip.getPositionInGround(state)
p_pelvis = osim.Vec3(0.5 * (p_left[0] + p_right[0]),
                     0.5 * (p_left[1] + p_right[1]),
                     0.5 * (p_left[2] + p_right[2]))

offset = ground.findStationLocationInAnotherFrame(state, p_pelvis, pelvis)
add_theia_frame(model, state, 'lower_pelvis', '/bodyset/pelvis', offset)

# Add torso frame.
# ----------------
# This frame is the located at the midpoint between the left and right shoulder joint,
# with an additional small vertical offset, expressed in the torso frame.
state = model.initSystem()
ground = model.getGround()
torso = osim.Body.safeDownCast(
        model.getComponent('/bodyset/torso'))
left_shoulder = osim.PhysicalOffsetFrame.safeDownCast(
        model.getComponent('/jointset/acromial_l/humerus_l_offset'))
right_shoulder = osim.PhysicalOffsetFrame.safeDownCast(
        model.getComponent('/jointset/acromial_r/humerus_r_offset'))

p_left = left_shoulder.getPositionInGround(state)
p_right = right_shoulder.getPositionInGround(state)
p_torso = osim.Vec3(0.5 * (p_left[0] + p_right[0]),
                    0.5 * (p_left[1] + p_right[1]),
                    0.5 * (p_left[2] + p_right[2]))

offset = ground.findStationLocationInAnotherFrame(state, p_torso, torso)
offset[1] += 0.05  # add 5 cm vertical offset
add_theia_frame(model, state, 'upper_torso', '/bodyset/torso', offset)

model.finalizeConnections()
model.printToXML('unscaled_generic.osim')
