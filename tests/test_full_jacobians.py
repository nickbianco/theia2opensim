import unittest
import numpy as np
import casadi as ca
import opensim as osim
from theia2opensim.callbacks import Callback, JacobianCallback
from theia2opensim.utilities import get_coordinate_indexes, calc_scaled_frame_position, \
                                    calc_scaled_frame_position_jacobian, \
                                    calc_scaled_frame_position_kinematic_jacobian

# Position Callbacks
# ------------------
class ScaledPositionMixin:
    def _init_scaled_position_components(self, model, frame_path):
        self.frame_path = frame_path
        self.frame = osim.PhysicalFrame.safeDownCast(
            model.getComponent(frame_path))
        self.base_frame_name = self.frame.findBaseFrame().getName()
        self.mobod_index = self.frame.getMobilizedBodyIndex()
        self.body_names = [body.getName() for body in model.getBodySet()]

    def _get_num_inputs(self):
        return len(self.coordinate_indexes) + 3*self.model.getBodySet().getSize()

    def _get_num_outputs(self):
        return 3

    def _apply_state(self, arg):
        # Apply the input coordinates to the model state and realize the system to the
        # position stage.
        q = np.zeros(self.state.getNQ())
        q[self.coordinate_indexes] = np.squeeze(arg[0][0:len(self.coordinate_indexes)].full())
        self.state.setQ(osim.Vector.createFromMat(q))
        self.model.realizePosition(self.state)

    def _expand_scale_factors(self, arg):
        scales = dict()
        scales_flat = np.squeeze(arg[0][len(self.coordinate_indexes):].full())
        for ibody, body_name in enumerate(self.body_names):
            scales[body_name] = np.array([scales_flat[3 * ibody],
                                          scales_flat[3 * ibody + 1],
                                          scales_flat[3 * ibody + 2]])
        scales['ground'] = np.ones(3)
        return scales

    def _eval(self, arg):
        self._apply_state(arg)
        scales = self._expand_scale_factors(arg)
        position = calc_scaled_frame_position(self.model,
                self.state, scales, self.frame)
        return [position]


class ScaledPositionCallback(ScaledPositionMixin, Callback):
    def __init__(self, name, model, coordinate_indexes, frame_path, opts={}):
        Callback.__init__(self, name, model, coordinate_indexes, opts)
        self._init_scaled_position_components(model, frame_path)


class ScaledPositionJacobianCallback(ScaledPositionMixin, JacobianCallback):
    def __init__(self, name, model, coordinate_indexes, frame_path, opts={}):
        JacobianCallback.__init__(self, name, model, coordinate_indexes, opts)
        self._init_scaled_position_components(model, frame_path)

    def _jac_eval(self, arg):
        # Kinematic Jacobian
        self._apply_state(arg)
        scales = self._expand_scale_factors(arg)
        matrix = osim.Matrix()
        station = self.frame.findTransformInBaseFrame().p()
        self.matter.calcStationJacobian(self.state, self.mobod_index, station,
                                        matrix)
        kinematic_jacobian = matrix.to_numpy()[:, self.coordinate_indexes]

        # TODO: this is needed for the kinematic Jacobian to be correct, but
        # we shouldn't need a hacky correction factor here.
        # kinematic_jacobian *= 2.0
        # kinematic_jacobian[:, 3:6] *= 0.5

        # Scale Jacobian
        scales = self._expand_scale_factors(arg)
        scale_jacobian_dict = calc_scaled_frame_position_jacobian(self.model,
                self.state, scales, self.frame)
        del scale_jacobian_dict['ground']

        scale_jacobian = np.zeros((3, 3*len(scale_jacobian_dict.keys())))
        for ibody, body_name in enumerate(scale_jacobian_dict.keys()):
            scale_jacobian[:, 3 * ibody] = scale_jacobian_dict[body_name][:, 0]
            scale_jacobian[:, 3 * ibody + 1] = scale_jacobian_dict[body_name][:, 1]
            scale_jacobian[:, 3 * ibody + 2] = scale_jacobian_dict[body_name][:, 2]

        return [np.concatenate([kinematic_jacobian, scale_jacobian], axis=1)]

# Unit tests
# ----------
# This needs to be generated via 'create_generic_model.py' script first.
MODEL_FPATH = 'unscaled_generic.osim'
FRAME_PATHS = [
               '/bodyset/pelvis/pelvis',
               '/bodyset/torso/torso',
               '/jointset/hip_r/femur_r_offset/r_thigh',
               '/jointset/walker_knee_r/tibia_r_offset/r_shank',
               '/jointset/ankle_r/talus_r_offset/r_foot',
               '/jointset/mtp_r/toes_r_offset/r_toes',
               '/bodyset/tibia_r'
               ]

class TestBodyScaleJacobians(unittest.TestCase):
    def test_body_scale_jacobians(self):
        model = osim.Model(MODEL_FPATH)
        state = model.initSystem()
        coordinates_map = get_coordinate_indexes(model, skip_dependent_coordinates=True)
        coordinate_indexes = list(coordinates_map.values())

        for frame_path in FRAME_PATHS:

            print(f'Frame path: {frame_path}')
            # Callback functions.
            f_fd = ScaledPositionCallback('f_fd', model, coordinate_indexes, frame_path,
                                          {"enable_fd": True})
            f_jac = ScaledPositionJacobianCallback('f_jac', model, coordinate_indexes,
                                                    frame_path)

            # Symbolic inputs.
            x = ca.SX.sym('x', len(coordinate_indexes) + 3*model.getBodySet().getSize())

            # Jacobian expression graphs.
            J_fd = ca.Function('J_fd',[x],[ca.jacobian(f_fd(x), x)])
            J_jac = ca.Function('J_jac',[x],[ca.jacobian(f_jac(x), x)])
            print(J_fd(2))
            print(J_jac(2))

            # element-wise divide the two Jacobians
            J_div = J_jac(2).full() / J_fd(2).full()

            # Jacobian components
            keys = list(coordinates_map.keys())
            for ikey, key in enumerate(keys):
                print(f'd pos / d {key}: {J_div[:, ikey]}')

            # Test that the two Jacobians are equivalent.
            self.assertTrue(np.allclose(J_jac(2).full(), J_fd(2).full(), atol=1e-6))


class TestCompareScaledAndUnscaledJacobians(unittest.TestCase):
    def test_compare_scaled_and_unscaled_jacobians(self):

        # Frame path to test.
        frame_path = FRAME_PATHS[1]

        # Create scale factors.
        scales = dict()
        scales['ground'] = np.ones(3) # Ground is not scaled.
        scales['pelvis'] = np.array([1.1, 1.2, 1.3])
        scales['femur_r'] = np.array([1.4, 1.5, 1.6])
        scales['tibia_r'] = np.array([1.7, 1.8, 1.9])

        def calc_unscaled_jacobian():
            model = osim.Model(MODEL_FPATH)
            state = model.initSystem()
            matter = model.getMatterSubsystem()
            frame = osim.PhysicalFrame.safeDownCast(
                model.getComponent(frame_path))
            mobod_index = frame.getMobilizedBodyIndex()

            matrix = osim.Matrix()
            station = frame.findTransformInBaseFrame().p()
            matter.calcSystemJacobian(state, matrix)
            jacobian = matrix.to_numpy()

            return jacobian

        def calc_scaled_jacobian():
            model = osim.Model(MODEL_FPATH)
            state = model.initSystem()

            # Create a scale for the right femur.
            scale_set = osim.ScaleSet()
            for body_name, scale_factors in scales.items():
                scale = osim.Scale()
                scale.setSegmentName(body_name)
                scale.setScaleFactors(
                    osim.Vec3(scale_factors[0], scale_factors[1], scale_factors[2]))
                scale_set.cloneAndAppend(scale)

            model.scale(state, scale_set, True)

            matter = model.getMatterSubsystem()
            frame = osim.PhysicalFrame.safeDownCast(
                model.getComponent(frame_path))
            mobod_index = frame.getMobilizedBodyIndex()

            matrix = osim.Matrix()
            station = frame.findTransformInBaseFrame().p()
            matter.calcSystemJacobian(state, matrix)
            jacobian = matrix.to_numpy()

            return jacobian

        unscaled_jacobian = calc_unscaled_jacobian()
        scaled_jacobian = calc_scaled_jacobian()
        ratio = scaled_jacobian / unscaled_jacobian

        # print non-nan values of ratio
        print(ratio[~np.isnan(ratio)])
        import pdb; pdb.set_trace()

        # print(unscaled_jacobian)
        # print(scaled_jacobian)
        # print(ratio)

        self.assertTrue(np.allclose(unscaled_jacobian, scaled_jacobian, atol=1e-6))
