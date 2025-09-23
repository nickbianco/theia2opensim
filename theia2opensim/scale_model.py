import os
import opensim as osim
import numpy as np
from theia2opensim.utilities import C3D

def compute_scale_factor(state, offset_frame_1, offset_frame_2,
                         frame_1_position, frame_2_position):
    # Magnitude of relative position between the two model frames.
    offset_frame_1_position = offset_frame_1.getPositionInGround(state).to_numpy()
    offset_frame_2_position = offset_frame_2.getPositionInGround(state).to_numpy()
    offset_frame_distance = np.linalg.norm(offset_frame_1_position -
                                           offset_frame_2_position)

    # Magnitude of relative position between the two Theia frames.
    theia_frame_distance = np.linalg.norm(frame_1_position.to_numpy() -
                                          frame_2_position.to_numpy())

    scale_factor = theia_frame_distance / offset_frame_distance
    return scale_factor


def create_scale(segment, scale_rules, offset_frame_map, positions, model, state):
    scale = osim.Scale()
    scale.setSegmentName(segment)
    factors = osim.Vec3(1.0)
    for rule in scale_rules:
        frame_1 = rule[0]
        frame_2 = rule[1]
        index = rule[2]
        offset_frame_1 = osim.PhysicalFrame.safeDownCast(
            model.getComponent(f'{offset_frame_map[frame_1]}/{frame_1}'))
        offset_frame_2 = osim.PhysicalFrame.safeDownCast(
            model.getComponent(f'{offset_frame_map[frame_2]}/{frame_2}'))
        frame_1_position = positions.getDependentColumn(frame_1)[0]
        frame_2_position = positions.getDependentColumn(frame_2)[0]
        factors[index] = compute_scale_factor(state, offset_frame_1, offset_frame_2,
                                              frame_1_position, frame_2_position)
    scale.setScaleFactors(factors)

    return scale


def scale_model(generic_model_fpath, trial_path, c3d_filename, offset_frame_map,
                scale_rules, enforce_symmetry, hip_joint_offset, scaled_model_name):

    # Load the model generic model.
    model = osim.Model(generic_model_fpath)
    state = model.initSystem()

    # Apply the hip joint offset.
    hip_r_pelvis_offset = osim.PhysicalOffsetFrame.safeDownCast(
        model.updComponent('/jointset/hip_r/pelvis_offset'))
    hip_r_pelvis_offset.upd_translation()[2] += hip_joint_offset
    hip_l_pelvis_offset = osim.PhysicalOffsetFrame.safeDownCast(
        model.updComponent('/jointset/hip_l/pelvis_offset'))
    hip_l_pelvis_offset.upd_translation()[2] -= hip_joint_offset
    model.finalizeConnections()
    model.initSystem()

    # Import the C3D file and load the frame position data.
    c3d = C3D(os.path.join(trial_path, c3d_filename))
    positions = c3d.get_positions_table()

    # Create scale factors
    # --------------------
    scaleset = osim.ScaleSet()
    for segment, rules in scale_rules.items():
        scale = create_scale(segment, rules, offset_frame_map, positions, model, state)
        scaleset.cloneAndAppend(scale)

    # Apply symmetry to scale factors.
    # --------------------------------
    if enforce_symmetry:
        for i in range(scaleset.getSize()):
            scale_r = scaleset.get(i)
            segment_name_r = scale.getSegmentName()
            if segment_name_r.endswith('_r'):
                segment_name_l = segment_name_r.replace('_r', '_l')
                for j in range(scaleset.getSize()):
                    scale_l = scaleset.get(j)
                    if scale_l.getSegmentName() == segment_name_l:
                        factors_r = scale_r.getScaleFactors()
                        factors_l = scale_l.getScaleFactors()
                        avg_factors = osim.Vec3(
                            0.5 * (factors_r[0] + factors_l[0]),
                            0.5 * (factors_r[1] + factors_l[1]),
                            0.5 * (factors_r[2] + factors_l[2]))
                        scale_r.setScaleFactors(avg_factors)
                        scale_l.setScaleFactors(avg_factors)

    # Scale the model
    # ---------------
    model.scale(state, scaleset, True)
    model.finalizeConnections()
    state = model.initSystem()
    model.setName(scaled_model_name)
    model.printToXML(os.path.join(trial_path, 'jump_1_scaled.osim'))
