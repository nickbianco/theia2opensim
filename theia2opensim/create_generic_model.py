import os
import numpy as np
import opensim as osim

def add_theia_frame(model, state, theia_frame_name, model_frame_path, offset):
    # Get the generic model frame and its transform in the ground frame.
    frame = osim.PhysicalFrame.safeDownCast(model.updComponent(model_frame_path))
    T_GO = frame.getTransformInGround(state)

    # Create an identity rotation representing the orientation of the new frame with
    # respect to the ground frame, R_GF = I.
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

def create_generic_model(model_fpath, offset_frame_map, torso_frame_offset,
                         generic_model_fpath):

    # Load the generic model and weld the subtalar joints.
    modelProcessor = osim.ModelProcessor(model_fpath)
    jointsToWeld = osim.StdVectorString()
    jointsToWeld.append('subtalar_l')
    jointsToWeld.append('subtalar_r')
    modelProcessor.append(osim.ModOpReplaceJointsWithWelds(jointsToWeld))
    model = modelProcessor.process()
    model.initSystem()

    # Update the should coordinate ranges.
    coordset = model.updCoordinateSet()
    for name in ['arm_flex', 'arm_add', 'arm_rot']:
        for side in ['l', 'r']:
            coord = coordset.get(f'{name}_{side}')
            coord.setRangeMin(-2.0*np.pi)
            coord.setRangeMax(2.0*np.pi)

    # The state represents the model in the default pose: the model is facing forward
    # along the X axis, and the Z axis is pointing to the right (Y is up). The frame
    # calculations below depend on the model being in this default pose. If the default
    # pose changes, it will change how the frames are added to the model.
    state = model.initSystem()

    # Add new frames to the model based on the Theia data.
    zero_offset = osim.Vec3(0)
    for theia_frame_name in offset_frame_map.keys():
        if '/bodyset' not in offset_frame_map[theia_frame_name]:
            add_theia_frame(model, state, theia_frame_name,
                            offset_frame_map[theia_frame_name], zero_offset)

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
    add_theia_frame(model, state, 'pelvis', offset_frame_map['pelvis'], offset)

    # Add torso frame.
    # ----------------
    # This frame is the located at the midpoint between the left and right shoulder
    # joint, with an additional small vertical offset, expressed in the torso frame.
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
    offset[1] += torso_frame_offset
    add_theia_frame(model, state, 'torso', offset_frame_map['torso'], offset)

    model.finalizeConnections()
    model.setName('unscaled_generic')
    model.printToXML(generic_model_fpath)
