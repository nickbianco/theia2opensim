import unittest
import numpy as np
import casadi as ca
import opensim as osim
from abc import ABC, abstractmethod
from theia2opensim.callbacks import Callback, JacobianCallback
from theia2opensim.utilities import get_coordinate_indexes

osim.Logger.setLevelString('error')

# Base Callback Classes
# ---------------------
class ScaleCallback(ca.Callback, ABC):
    def __init__(self, name, model, opts={}):
        ca.Callback.__init__(self)
        self.model = model
        self.state = self.model.initSystem()
        self.matter = self.model.getMatterSubsystem()

        # Create the scale set.
        bodyset = self.model.getBodySet()
        self.num_scale_factors = 3 * bodyset.getSize()
        self.scale_set = osim.ScaleSet()
        for ibody in range(bodyset.getSize()):
            body = bodyset.get(ibody)
            body_name = body.getName()
            scale = osim.Scale()
            scale.setSegmentName(body_name)
            scale.setScaleFactors(osim.Vec3(1.0))
            self.scale_set.cloneAndAppend(scale)

        self.construct(name, opts)

    def get_num_inputs(self):
        return self._get_num_inputs()
    def get_num_outputs(self):
        return self._get_num_outputs()

    def get_n_in(self): return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self,i):
        return ca.Sparsity.dense(self.get_num_inputs(), 1)
    def get_sparsity_out(self,i):
        return ca.Sparsity.dense(self.get_num_outputs(), 1)

    def eval(self, arg):
        return self._eval(arg)

    def apply_scale_factors(self, arg):
        # Apply the input scale factors to the model.
        scale_factors = np.squeeze(arg[0].full())
        for i in range(self.scale_set.getSize()):
            scale = self.scale_set.get(i)
            scale.setScaleFactors(osim.Vec3(scale_factors[i*3],
                                            scale_factors[i*3+1],
                                            scale_factors[i*3+2]))


    @abstractmethod
    def _get_num_inputs(self):
        pass

    @abstractmethod
    def _get_num_outputs(self):
        pass

    @abstractmethod
    def _eval(self, arg):
        pass


class ScaleJacobianCallback(ScaleCallback, ABC):
    def has_jacobian(self): return True
    def get_jacobian(self, name, inames, onames, opts):
        class JacobianFunction(ca.Callback):
            def __init__(self, callback, opts={}):
                ca.Callback.__init__(self)
                self.callback = callback
                self.construct(name, opts)

            def get_n_in(self): return 2
            def get_n_out(self): return 1

            def get_sparsity_in(self,i):
                if i == 0:
                    return ca.Sparsity.dense(self.callback.get_num_inputs(), 1)
                elif i == 1:
                    return ca.Sparsity.dense(self.callback.get_num_outputs(), 1)
            def get_sparsity_out(self,i):
                return ca.Sparsity.dense(self.callback.get_num_outputs(),
                                         self.callback.get_num_inputs())

            def eval(self, arg):
                return self.callback._jac_eval(arg)

        self.jacobian_callback = JacobianFunction(self)
        return self.jacobian_callback

    @abstractmethod
    def _jac_eval(self, arg):
        pass

# Body Scale Callbacks
# --------------------
class BodyScaleMixin:
    def _init_scale_components(self, model, body_name):
        self.body = model.getBodySet().get(body_name)
        self.mobod_index = self.body.getMobilizedBodyIndex()

    def _get_num_inputs(self):
        return self.num_scale_factors

    def _get_num_outputs(self):
        return 3

    def _eval(self, arg):
        self.apply_scale_factors(arg)
        position = self.body.getPositionInGround(self.state).to_numpy()
        return [position]


class BodyScaleCallback(BodyScaleMixin, ScaleCallback):
    def __init__(self, name, model, body_name, opts={}):
        ScaleCallback.__init__(self, name, model, opts)
        self._init_scale_components(model, body_name)


class BodyScaleJacobianCallback(BodyScaleMixin, ScaleJacobianCallback):
    def __init__(self, name, model, body_name, opts={}):
        ScaleJacobianCallback.__init__(self, name, model, opts)
        self._init_scale_components(model, body_name)

    def _jac_eval(self, arg):
        self.apply_scale_factors(arg)
        return [np.random.rand(3, self.get_num_inputs())]



# Unit tests
# ----------
# This needs to be generated via 'create_generic_model.py' script first.
MODEL_FPATH = 'unscaled_generic.osim'
FRAME_PATHS = [
               '/bodyset/pelvis/pelvis',
               '/bodyset/torso/torso',
               '/jointset/hip_r/femur_r_offset',
               '/jointset/hip_r/femur_r_offset/r_thigh',
               '/bodyset/tibia_r',
               '/jointset/walker_knee_r/tibia_r_offset',
               '/jointset/walker_knee_r/tibia_r_offset/r_shank',
               '/jointset/ankle_r/talus_r_offset/r_foot',
               '/jointset/mtp_r/toes_r_offset/r_toes'
               ]


def rotation_to_numpy(rotation):
    rot = np.zeros((3, 3))
    rot[0,0] = rotation.get(0, 0)
    rot[0,1] = rotation.get(0, 1)
    rot[0,2] = rotation.get(0, 2)
    rot[1,0] = rotation.get(1, 0)
    rot[1,1] = rotation.get(1, 1)
    rot[1,2] = rotation.get(1, 2)
    rot[2,0] = rotation.get(2, 0)
    rot[2,1] = rotation.get(2, 1)
    rot[2,2] = rotation.get(2, 2)
    return rot


def calc_positions(model, state):
    positions = dict()
    mobilizer_translations = dict()
    rotations = dict()
    positions_in_base = dict()

    ground = model.getGround()
    positions['ground'] = ground.getPositionInGround(state).to_numpy()
    rotations['ground'] = rotation_to_numpy(ground.getRotationInGround(state))

    bodyset = model.getBodySet()
    for ibody in range(bodyset.getSize()):
        body = bodyset.get(ibody)
        body_name = body.getName()
        positions[body_name] = body.getPositionInGround(state).to_numpy()
        rotations[body_name] = rotation_to_numpy(body.getRotationInGround(state))

    jointset = model.getJointSet()
    for ijoint in range(jointset.getSize()):
        joint = jointset.get(ijoint)
        joint_name = joint.getName()
        parent_offset = osim.PhysicalFrame.safeDownCast(joint.getParentFrame())
        parent = osim.PhysicalFrame.safeDownCast(parent_offset.findBaseFrame())
        child_offset = osim.PhysicalFrame.safeDownCast(joint.getChildFrame())
        child = osim.PhysicalFrame.safeDownCast(child_offset.findBaseFrame())

        mobilizer_translations[joint_name] = \
            child.getMobilizerTransform(state).p().to_numpy()

        parent_offset_name = parent_offset.getName()
        child_offset_name = child_offset.getName()
        positions[joint_name] = {
            parent_offset_name: parent_offset.getPositionInGround(state).to_numpy(),
            child_offset_name: child_offset.getPositionInGround(state).to_numpy()
        }
        positions_in_base[joint_name] = {
            parent_offset_name: parent_offset.findTransformInBaseFrame().p().to_numpy(),
            child_offset_name: child_offset.findTransformInBaseFrame().p().to_numpy()
        }
        rotations[joint_name] = {
            parent_offset_name: rotation_to_numpy(parent_offset.getRotationInGround(state)),
            child_offset_name: rotation_to_numpy(child_offset.getRotationInGround(state))
        }

    return positions, rotations, positions_in_base, mobilizer_translations


parents = {
    'ground_pelvis': 'ground',
    'hip_r': 'pelvis',
    'walker_knee_r': 'femur_r',
}
children = {
    'ground_pelvis': 'pelvis',
    'hip_r': 'femur_r',
    'walker_knee_r': 'tibia_r',
}
parent_offsets = {
    'ground_pelvis': 'ground_offset',
    'hip_r': 'pelvis_offset',
    'walker_knee_r': 'femur_r_offset',
}
child_offsets = {
    'ground_pelvis': 'pelvis_offset',
    'hip_r': 'femur_r_offset',
    'walker_knee_r': 'tibia_r_offset',
}

def forward_kinematics(model, state, scales, frame_path, frame_tree):

    positions, rotations, positions_in_base, mobilizer_translations = \
        calc_positions(model, state)

    frame = osim.Frame.safeDownCast(model.getComponent(frame_path))

    # Initialize the frame position.
    frame_position = np.zeros(3)
    frame_position += positions['ground']

    # Iterate through the frames's joint tree.
    for joint_name in frame_tree:
        parent_name = parents[joint_name]
        child_name = children[joint_name]
        parent_offset_name = parent_offsets[joint_name]
        child_offset_name = child_offsets[joint_name]

        # Contribution from the parent offset frame.
        frame_position += np.dot(rotations[parent_name],
                np.multiply(positions_in_base[joint_name][parent_offset_name],
                            scales[parent_name]))

        # Contribution from the mobilizer translation.
        frame_position += np.dot(
                rotations[joint_name][parent_offset_name],
                np.multiply(mobilizer_translations[joint_name], scales[parent_name]))

        # Contribution from the child offset frame.
        frame_position += np.dot(rotations[child_name],
                np.multiply(-positions_in_base[joint_name][child_offset_name],
                            scales[child_name]))

    # Contribution from the frame offset in the child body of the final joint.
    frame_offset = frame.findTransformInBaseFrame().p().to_numpy()
    frame_position += np.dot(
        rotations[children[frame_tree[-1]]],
        np.multiply(frame_offset, scales[children[frame_tree[-1]]]))

    return frame_position

class TestCustomJointScale(unittest.TestCase):
    def test_custom_joint_scale(self):
        model = osim.Model(MODEL_FPATH)
        state = model.initSystem()
        model.realizePosition(state)

        # Create scale factors.
        scales = dict()
        scales['ground'] = np.ones(3) # Ground is not scaled.
        scales['pelvis'] = np.array([1.1, 1.2, 1.3])
        scales['femur_r'] = np.array([1.4, 1.5, 1.6])
        scales['tibia_r'] = np.array([1.7, 1.8, 1.9])

        positions_pre, rotations_pre, positions_in_base_pre, mobilizer_translations_pre = \
            calc_positions(model, state)

        # Create a scale for the right femur.
        scale_set = osim.ScaleSet()
        for body_name, scale_factors in scales.items():
            scale = osim.Scale()
            scale.setSegmentName(body_name)
            scale.setScaleFactors(osim.Vec3(scale_factors[0], scale_factors[1], scale_factors[2]))
            scale_set.cloneAndAppend(scale)

        # Scale the model.
        state = model.initSystem()
        model.scale(state, scale_set, True)
        state = model.initSystem()
        model.realizePosition(state)

        positions_post, _, _, _ = calc_positions(model, state)

        ground_offset_position_post = positions_pre['ground'] + \
            positions_in_base_pre['ground_pelvis']['ground_offset']
        self.assertTrue(np.allclose(ground_offset_position_post,
            positions_post['ground_pelvis']['ground_offset'], atol=1e-6))

        pelvis_offset_position_post = ground_offset_position_post + \
            np.dot(rotations_pre['ground_pelvis']['ground_offset'],
                   mobilizer_translations_pre['ground_pelvis'])
        self.assertTrue(np.allclose(pelvis_offset_position_post,
            positions_post['ground_pelvis']['pelvis_offset'], atol=1e-6))

        pelvis_position_post = pelvis_offset_position_post + \
            np.dot(rotations_pre['pelvis'],
                   np.multiply(-positions_in_base_pre['ground_pelvis']['pelvis_offset'], scales['pelvis']))
        self.assertTrue(np.allclose(pelvis_position_post,
            positions_post['pelvis'], atol=1e-6))

        pelvis_offset_position_post = pelvis_position_post + \
            np.dot(rotations_pre['pelvis'],
                   np.multiply(positions_in_base_pre['hip_r']['pelvis_offset'], scales['pelvis']))
        self.assertTrue(np.allclose(pelvis_offset_position_post,
            positions_post['hip_r']['pelvis_offset'], atol=1e-6))

        femur_r_offset_position_post = pelvis_offset_position_post + \
            np.dot(rotations_pre['hip_r']['pelvis_offset'],
                   np.multiply(mobilizer_translations_pre['hip_r'], scales['pelvis']))
        self.assertTrue(np.allclose(femur_r_offset_position_post,
            positions_post['hip_r']['femur_r_offset'], atol=1e-6))

        femur_r_position_post = femur_r_offset_position_post + \
            np.dot(rotations_pre['femur_r'],
                   np.multiply(-positions_in_base_pre['hip_r']['femur_r_offset'], scales['femur_r']))
        self.assertTrue(np.allclose(femur_r_position_post,
            positions_post['femur_r'], atol=1e-6))

        femur_r_offset_position_post = femur_r_position_post + \
            np.dot(rotations_pre['femur_r'],
                   np.multiply(positions_in_base_pre['walker_knee_r']['femur_r_offset'], scales['femur_r']))
        self.assertTrue(np.allclose(femur_r_offset_position_post,
            positions_post['walker_knee_r']['femur_r_offset'], atol=1e-6))

        tibia_r_offset_position_post = femur_r_offset_position_post + \
            np.dot(rotations_pre['walker_knee_r']['femur_r_offset'],
                   np.multiply(mobilizer_translations_pre['walker_knee_r'], scales['femur_r']))
        self.assertTrue(np.allclose(tibia_r_offset_position_post,
            positions_post['walker_knee_r']['tibia_r_offset'], atol=1e-6))

        tibia_r_position_post = tibia_r_offset_position_post + \
            np.dot(rotations_pre['tibia_r'],
                   np.multiply(-positions_in_base_pre['walker_knee_r']['tibia_r_offset'], scales['tibia_r']))
        self.assertTrue(np.allclose(tibia_r_position_post,
            positions_post['tibia_r'], atol=1e-6))


class TestCalculateFramePosition(unittest.TestCase):
    def test_calculate_frame_position(self):
        model = osim.Model(MODEL_FPATH)
        state = model.initSystem()
        model.realizePosition(state)

         # Create scale factors.
        scales = dict()
        scales['ground'] = np.ones(3) # Ground is not scaled.
        scales['pelvis'] = np.array([1.1, 1.2, 1.3])
        scales['femur_r'] = np.array([1.4, 1.5, 1.6])
        scales['tibia_r'] = np.array([1.7, 1.8, 1.9])

        # Calculate the frame position using the forward kinematics function.
        frame_path = '/jointset/walker_knee_r/tibia_r_offset/r_shank'
        frame_tree = ['ground_pelvis', 'hip_r', 'walker_knee_r']
        frame_position = forward_kinematics(model, state, scales, frame_path, frame_tree)

        # Create a scale for the right femur.
        scale_set = osim.ScaleSet()
        for body_name, scale_factors in scales.items():
            scale = osim.Scale()
            scale.setSegmentName(body_name)
            scale.setScaleFactors(
                osim.Vec3(scale_factors[0], scale_factors[1], scale_factors[2]))
            scale_set.cloneAndAppend(scale)

        # Scale the model.
        state = model.initSystem()
        model.scale(state, scale_set, True)
        state = model.initSystem()
        model.realizePosition(state)

        # Compare the frame position to the frame position from OpenSim after scaling.
        frame = osim.Frame.safeDownCast(model.getComponent(frame_path))
        self.assertTrue(np.allclose(frame_position,
            frame.getPositionInGround(state).to_numpy(), atol=1e-6))


# class TestScaleEquivalence(unittest.TestCase):
#     def test_scale_equivalence(self):

#         model = osim.Model(MODEL_FPATH)
#         model.initSystem()

#         # Create random scale factors.
#         num_scale_factors = 3 * model.getBodySet().getSize()
#         scale_factors = np.ones(num_scale_factors)
#         body_indexes = dict()
#         for ibody in range(model.getBodySet().getSize()):
#             body = model.getBodySet().get(ibody)
#             body_name = body.getName()
#             body_indexes[body_name] = ibody

#         # Create an OpenSim ScaleSet from scale factors.
#         def create_scale_set(scale_factors):
#             scale_set = osim.ScaleSet()
#             for ibody in range(model.getBodySet().getSize()):
#                 body = model.getBodySet().get(ibody)
#                 body_name = body.getName()
#                 scale = osim.Scale()
#                 scale.setSegmentName(body_name)
#                 scale.setScaleFactors(osim.Vec3(scale_factors[ibody*3],
#                                                 scale_factors[ibody*3+1],
#                                                 scale_factors[ibody*3+2]))
#                 scale_set.cloneAndAppend(scale)

#             return scale_set

#         # Forward kinematics using OpenSim's scale function.
#         def osim_forward_kinematics(frame_path, scale_factors):
#             model = osim.Model(MODEL_FPATH)
#             state =model.initSystem()
#             scale_set = create_scale_set(scale_factors)
#             model.scale(state, scale_set, True)
#             state = model.initSystem()
#             model.realizePosition(state)

#             frame = osim.Frame.safeDownCast(model.getComponent(frame_path))
#             return frame.getPositionInGround(state).to_numpy()


#         def find_joint_parent_frame(child_base_frame):
#             jointset = model.getJointSet()
#             for ijoint in range(jointset.getSize()):
#                 joint = jointset.get(ijoint)
#                 joint_child_base_frame = joint.getChildFrame().findBaseFrame()
#                 if joint_child_base_frame.getName() == child_base_frame.getName():
#                     return joint.getParentFrame()
#             return None

#         def forward_kinematics(frame_path, scale_factors):

#             # Initialize the model and state.
#             model = osim.Model(MODEL_FPATH)
#             state = model.initSystem()

#             # Initialize the frame position.
#             frame_position = np.zeros(3)

#             # Create the scale set.
#             scale_set = create_scale_set(scale_factors)

#             # Get the end-effector frame and its base frame.
#             child_frame = model.getComponent(frame_path)
#             child_base_frame = osim.PhysicalFrame.safeDownCast(
#                 child_frame.findBaseFrame())

#             # Compute the scaled position of the end-effector frame in its base frame,
#             # expressed in ground. This quantity is added to the total frame position
#             # vector.
#             child_position_in_base = child_frame.findTransformInBaseFrame().p().to_numpy()
#             child_base_rotation_in_ground = rotation_to_numpy(
#                 child_base_frame.getRotationInGround(state))
#             scale_factors = scale_set.get(
#                 body_indexes[child_base_frame.getName()]).getScaleFactors().to_numpy()
#             frame_position += np.dot(child_base_rotation_in_ground,
#                 np.multiply(scale_factors, child_position_in_base))

#             # Recurse through the parent frames until we hit the ground frame to compute
#             # the total frame position.
#             while True:
#                 # Get the parent frame of the joint that mobilizes the child frame.
#                 # Also, get the base frame of the parent frame.
#                 parent_frame = find_joint_parent_frame(child_base_frame)
#                 parent_base_frame = osim.PhysicalFrame.safeDownCast(
#                     parent_frame.findBaseFrame())

#                 # Get the mobilizer translation. This is expressed in the parent frame.
#                 mobilizer_transform = child_base_frame.getMobilizerTransform(state)
#                 mobilizer_translation = mobilizer_transform.p().to_numpy()

#                 # Get the transform of the parent frame in its base frame.
#                 parent_transform_in_base = parent_frame.findTransformInBaseFrame()
#                 parent_position_in_base = parent_transform_in_base.p().to_numpy()

#                 # Get the rotation of the parent frame in its base frame.
#                 parent_base_rotation_in_ground = rotation_to_numpy(
#                     parent_base_frame.getRotationInGround(state))
#                 parent_rotation_in_ground = rotation_to_numpy(
#                     parent_frame.getRotationInGround(state))

#                 if parent_base_frame.getName() == 'ground':
#                     # Compute the contribution of the mobilizer to the frame position.
#                     frame_position += np.dot(parent_rotation_in_ground,
#                                              mobilizer_translation)

#                     # Compute the contribution of the parent frame to the frame position.
#                     frame_position += np.dot(parent_base_rotation_in_ground,
#                                              parent_position_in_base)
#                     break

#                 # Get the scale factors for the parent frame.
#                 parent_frame_name = parent_base_frame.getName()
#                 scale_factors = scale_set.get(
#                     body_indexes[parent_frame_name]).getScaleFactors().to_numpy()

#                 # Compute the contribution of the mobilizer to the frame position.
#                 frame_position += np.dot(
#                     parent_rotation_in_ground,
#                     np.multiply(mobilizer_translation, scale_factors))

#                 # Compute the contribution of the parent frame to the frame position.
#                 frame_position += np.dot(
#                     parent_base_rotation_in_ground,
#                     np.multiply(parent_position_in_base, scale_factors))

#                 child_base_frame = parent_base_frame

#             return frame_position

#         # Jacobian of forward kinematics with respect to scale factors using
#         # for a single frame.
#         def jacobian_forward_kinematics(fk_func, frame_path, scale_factors):
#             J = np.zeros((3, num_scale_factors))
#             eps = 1e-5
#             for i in range(num_scale_factors):
#                 scale_factors_plus = scale_factors.copy()
#                 scale_factors_plus[i] += eps
#                 scale_factors_minus = scale_factors.copy()
#                 scale_factors_minus[i] -= eps
#                 frame_positions_plus = fk_func(frame_path, scale_factors_plus)
#                 frame_positions_minus = fk_func(frame_path, scale_factors_minus)
#                 J[:,i] = (frame_positions_plus - frame_positions_minus) / (2 * eps)
#             return J

#         for frame_path in FRAME_PATHS:
#             print(f'Frame path: {frame_path}')
#             osim_frame_position = osim_forward_kinematics(frame_path, scale_factors)
#             forward_frame_position = forward_kinematics(frame_path, scale_factors)
#             print(f'OSIM frame position: {osim_frame_position}')
#             print(f'Forward frame position: {forward_frame_position}')
#             print(f'Difference: {osim_frame_position - forward_frame_position}')
#             print('')
#             self.assertTrue(np.allclose(osim_frame_position, forward_frame_position, atol=1e-6))

# class TestBodyScaleJacobians(unittest.TestCase):
#     def test_body_scale_jacobians(self):
#         model = osim.Model(MODEL_FPATH)
#         state = model.initSystem()

#         bodyset = model.getBodySet()

#         pelvis = bodyset.get('pelvis')

#         print('position in ground: ', pelvis.getPositionInGround(state))
#         print('position in base frame: ', pelvis.findTransformInBaseFrame().p())

#         # for ibody in range(bodyset.getSize()):
#         #     body = bodyset.get(ibody)
#         #     body_name = body.getName()

#         # Callback functions.
#         # f_fd = BodyScaleCallback('f_fd', model, 'torso', {"enable_fd": True})
#         # f_jac = BodyScaleJacobianCallback('f_jac', model, 'torso')

#         # # Symbolic inputs.
#         # x = ca.SX.sym('x', 3 * bodyset.getSize())

#         # # Jacobian expression graphs.
#         # J_fd = ca.Function('J_fd',[x],[ca.jacobian(f_fd(x), x)])
#         # J_jac = ca.Function('J_jac',[x],[ca.jacobian(f_jac(x), x)])
#         # print(J_fd(2))
#         # print(J_jac(2))

#         # Test that the two Jacobians are equivalent.
#         # self.assertTrue(np.allclose(J_jac(2).full(), J_fd(2).full(), atol=1e-6))


