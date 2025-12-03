import unittest
import numpy as np
import casadi as ca
import opensim as osim
from abc import ABC, abstractmethod
from theia2opensim.utilities import calc_scaled_frame_position, \
                                    calc_scaled_frame_position_jacobian

# Base Callback Classes
# ---------------------
class ScaleCallback(ca.Callback, ABC):
    def __init__(self, name, model, opts={}):
        ca.Callback.__init__(self)
        self.model = model
        self.state = self.model.initSystem()
        self.matter = self.model.getMatterSubsystem()
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

# Position Callbacks
# ------------------
class ScaledPositionMixin:
    def _init_scaled_position_components(self, model, frame_path):
        self.frame_path = frame_path
        self.frame = osim.PhysicalFrame.safeDownCast(
            model.getComponent(frame_path))
        self.mobod_index = self.frame.getMobilizedBodyIndex()
        self.body_names = [body.getName() for body in model.getBodySet()]

    def _get_num_inputs(self):
        return 3*self.model.getBodySet().getSize()

    def _get_num_outputs(self):
        return 3

    def _expand_scale_factors(self, arg):
        scales = dict()
        scales_flat = np.squeeze(arg[0].full())
        for ibody, body_name in enumerate(self.body_names):
            scales[body_name] = np.array([scales_flat[3 * ibody],
                                          scales_flat[3 * ibody + 1],
                                          scales_flat[3 * ibody + 2]])
        scales['ground'] = np.ones(3)
        return scales

    def _eval(self, arg):
        scales = self._expand_scale_factors(arg)
        position = calc_scaled_frame_position(self.model,
                self.state, scales, self.frame)
        return [position]


class ScaledPositionCallback(ScaledPositionMixin, ScaleCallback):
    def __init__(self, name, model, frame_path, opts={}):
        ScaleCallback.__init__(self, name, model, opts)
        self._init_scaled_position_components(model, frame_path)


class ScaledPositionJacobianCallback(ScaledPositionMixin, ScaleJacobianCallback):
    def __init__(self, name, model, frame_path, opts={}):
        ScaleJacobianCallback.__init__(self, name, model, opts)
        self._init_scaled_position_components(model, frame_path)

    def _jac_eval(self, arg):
        scales = self._expand_scale_factors(arg)
        jacobian_dict = calc_scaled_frame_position_jacobian(self.model,
                self.state, scales, self.frame)
        del jacobian_dict['ground']

        jacobian = np.zeros((3, 3*len(jacobian_dict.keys())))
        for ibody, body_name in enumerate(jacobian_dict.keys()):
            jacobian[:, 3 * ibody] = jacobian_dict[body_name][:, 0]
            jacobian[:, 3 * ibody + 1] = jacobian_dict[body_name][:, 1]
            jacobian[:, 3 * ibody + 2] = jacobian_dict[body_name][:, 2]

        return [jacobian]

# Unit tests
# ----------
# This needs to be generated via 'create_generic_model.py' script first.
MODEL_FPATH = 'unscaled_generic.osim'
FRAME_PATHS = ['/bodyset/pelvis/pelvis',
               '/bodyset/torso/torso',
               '/jointset/hip_r/femur_r_offset/r_thigh',
               '/jointset/walker_knee_r/tibia_r_offset/r_shank',
               '/jointset/ankle_r/talus_r_offset/r_foot',
               '/jointset/mtp_r/toes_r_offset/r_toes']

class TestCalculateScaledFramePosition(unittest.TestCase):
    def test_calculate_scaled_frame_position(self):
        model = osim.Model(MODEL_FPATH)
        state = model.initSystem()
        model.realizePosition(state)

         # Create scale factors.
        scales = dict()
        scales['ground'] = np.ones(3) # Ground is not scaled.
        scales['pelvis'] = np.array([1.1, 1.2, 1.3])
        scales['femur_r'] = np.array([1.4, 1.5, 1.6])
        scales['tibia_r'] = np.array([1.7, 1.8, 1.9])

        # Calculate the frame position as a function of the scale factors.
        frame_path = '/jointset/walker_knee_r/tibia_r_offset/r_shank'
        frame = osim.PhysicalFrame.safeDownCast(model.getComponent(frame_path))
        frame_position = calc_scaled_frame_position(model, state, scales, frame)

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


class TestBodyScaleJacobians(unittest.TestCase):
    def test_body_scale_jacobians(self):
        model = osim.Model(MODEL_FPATH)
        state = model.initSystem()

        for frame_path in FRAME_PATHS:
            # Callback functions.
            f_fd = ScaledPositionCallback('f_fd', model, frame_path,
                                          {"enable_fd": True})
            f_jac = ScaledPositionJacobianCallback('f_jac', model, frame_path)

            # Symbolic inputs.
            x = ca.SX.sym('x', 3*model.getBodySet().getSize())

            # Jacobian expression graphs.
            J_fd = ca.Function('J_fd',[x],[ca.jacobian(f_fd(x), x)])
            J_jac = ca.Function('J_jac',[x],[ca.jacobian(f_jac(x), x)])

            # Test that the two Jacobians are equivalent.
            self.assertTrue(np.allclose(J_jac(2).full(), J_fd(2).full(), atol=1e-6))
