import unittest
import numpy as np
import casadi as ca
import opensim as osim
from abc import ABC, abstractmethod

class Callback(ca.Callback, ABC):
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

class JacobianCallback(Callback, ABC):
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


# Position Jacobians
# ------------------
class PositionCallback(Callback):
    def __init__(self, name, model, body, opts={}):
        Callback.__init__(self, name, model, opts)
        self.body = self.model.getBodySet().get(body)
        self.mobod_index = self.body.getMobilizedBodyIndex()

    def _get_num_inputs(self):
        return self.state.getNQ()

    def _get_num_outputs(self):
        return 3

    def _eval(self, arg):
        self.state.setQ(osim.Vector.createFromMat(np.squeeze(arg[0].full())))
        self.model.realizePosition(self.state)
        position = self.body.getPositionInGround(self.state).to_numpy()
        return [position]


class PositionJacobianCallback(JacobianCallback):
    def __init__(self, name, model, body, opts={}):
        JacobianCallback.__init__(self, name, model, opts)
        self.body = self.model.getBodySet().get(body)
        self.mobod_index = self.body.getMobilizedBodyIndex()

    def _get_num_inputs(self):
        return self.state.getNQ()

    def _get_num_outputs(self):
        return 3

    def _eval(self, arg):
        self.state.setQ(osim.Vector.createFromMat(np.squeeze(arg[0].full())))
        self.model.realizePosition(self.state)
        position = self.body.getPositionInGround(self.state).to_numpy()
        return [position]

    def _jac_eval(self, arg):
        self.state.setQ(osim.Vector.createFromMat(np.squeeze(arg[0].full())))
        self.model.realizePosition(self.state)

        matrix = osim.Matrix()
        self.matter.calcStationJacobian(self.state, self.mobod_index, osim.Vec3(0),
                                        matrix)
        return [matrix.to_numpy()]


# class TestPositionJacobians(unittest.TestCase):
#     def test_position_jacobians(self):
#         model = osim.Model('unscaled_generic.osim')
#         state = model.initSystem()

#         bodyset = model.getBodySet()
#         for ibody in range(bodyset.getSize()):
#             body = bodyset.get(ibody)
#             body_name = body.getName()

#             # Callback functions.
#             f_fd = PositionCallback('f_fd', model, body_name, {"enable_fd": True})
#             f_jac = PositionJacobianCallback('f_jac', model, body_name)

#             # Symbolic inputs.
#             x = ca.MX.sym("x", state.getNQ())

#             # Jacobian expression graphs.
#             J_fd = ca.Function('J',[x],[ca.jacobian(f_fd(x), x)])
#             J_jac = ca.Function('J',[x],[ca.jacobian(f_jac(x), x)])

#             # Test that the two Jacobians are equivalent.
#             self.assertTrue(np.allclose(J_jac(2).full(), J_fd(2).full(),
#                                         atol=1e-6))


# Orientation Jacobians
# ---------------------
class OrientationCallback(Callback):
    def __init__(self, name, model, body, opts={}):
        Callback.__init__(self, name, model, opts)
        self.body = self.model.getBodySet().get(body)
        self.mobod_index = self.body.getMobilizedBodyIndex()

    def _get_num_inputs(self):
        return self.state.getNQ()

    def _get_num_outputs(self):
        return 4

    def _eval(self, arg):
        self.state.setQ(osim.Vector.createFromMat(np.squeeze(arg[0].full())))
        self.model.realizePosition(self.state)
        rotation = self.body.getRotationInGround(self.state)
        quaternion = rotation.convertRotationToQuaternion()
        return [np.array([quaternion.get(0), quaternion.get(1),
                          quaternion.get(2), quaternion.get(3)])]


class OrientationJacobianCallback(JacobianCallback):
    def __init__(self, name, model, body, opts={}):
        JacobianCallback.__init__(self, name, model, opts)
        self.body = self.model.getBodySet().get(body)
        self.mobod_index = self.body.getMobilizedBodyIndex()

    def _get_num_inputs(self):
        return self.state.getNQ()

    def _get_num_outputs(self):
        return 4

    def _calc_quaternion(self, arg):
        self.state.setQ(osim.Vector.createFromMat(np.squeeze(arg[0].full())))
        self.model.realizePosition(self.state)
        rotation = self.body.getRotationInGround(self.state)
        quaternion = rotation.convertRotationToQuaternion()
        eps = np.array([quaternion.get(0), quaternion.get(1),
                        quaternion.get(2), quaternion.get(3)])
        return eps

    def _eval(self, arg):
        eps = self._calc_quaternion(arg)
        return [eps]

    def _calc_quaternion_jacobian(self, arg):
        eps = self._calc_quaternion(arg)
        # [eps_dot; vdot] = J_eps * [omega; vdot]
        # J = [ -0.5*e1 -0.5*e2 -0.5*e3  0   0   0
        #        0.5*e0 -0.5*e3  0.5*e2  0   0   0
        #        0.5*e3  0.5*e0 -0.5*e1  0   0   0
        #       -0.5*e2  0.5*e1  0.5*e0  0   0   0
        #             0       0       0  1   0   0
        #             0       0       0  0   1   0
        #             0       0       0  0   0   1]
        jac_eps = osim.Matrix(7, 6, 0.0)
        jac_eps.set(0, 0, -0.5*eps[1])
        jac_eps.set(0, 1, -0.5*eps[2])
        jac_eps.set(0, 2, -0.5*eps[3])
        jac_eps.set(1, 0,  0.5*eps[0])
        jac_eps.set(1, 1, -0.5*eps[3])
        jac_eps.set(1, 2,  0.5*eps[2])
        jac_eps.set(2, 0,  0.5*eps[3])
        jac_eps.set(2, 1,  0.5*eps[0])
        jac_eps.set(2, 2, -0.5*eps[1])
        jac_eps.set(3, 0, -0.5*eps[2])
        jac_eps.set(3, 1,  0.5*eps[1])
        jac_eps.set(3, 2,  0.5*eps[0])
        jac_eps.set(4, 3,  1.0)
        jac_eps.set(5, 4,  1.0)
        jac_eps.set(6, 5,  1.0)
        return jac_eps.to_numpy()

    def _calc_frame_jacobian(self, arg):
        self.state.setQ(osim.Vector.createFromMat(np.squeeze(arg[0].full())))
        self.model.realizePosition(self.state)
        jac_frame = osim.Matrix()
        self.matter.calcFrameJacobian(self.state, self.mobod_index, osim.Vec3(0),
                                      jac_frame)
        return jac_frame.to_numpy()

    def _jac_eval(self, arg):
        jac_eps = self._calc_quaternion_jacobian(arg)
        jac_frame = self._calc_frame_jacobian(arg)
        jac = jac_eps.dot(jac_frame)
        return [jac[0:4, :]]


class TestOrientationJacobians(unittest.TestCase):
    def test_orientation_jacobians(self):
        model = osim.Model('unscaled_generic.osim')
        state = model.initSystem()

        # bodyset = model.getBodySet()
        # for ibody in range(bodyset.getSize()):
        #     body = bodyset.get(ibody)
        #     body_name = body.getName()

        # Callback functions.
        f_fd = OrientationCallback('f_fd', model, 'pelvis', {'enable_fd': True})
        f_jac = OrientationJacobianCallback('f_jac', model, 'pelvis')

        # Symbolic inputs.
        x = ca.MX.sym('x', state.getNQ())
        print(f_fd(x))

        # Jacobian expression graphs.
        J_fd = ca.Function('J',[x],[ca.jacobian(f_fd(x), x)])
        print(J_fd(2))
        J_jac = ca.Function('J',[x],[ca.jacobian(f_jac(x), x)])
        print(J_jac(2))

            # Test that the two Jacobians are equivalent.
            # self.assertTrue(np.allclose(J_jac(2).full(), J_fd(2).full(),
            #                             atol=1e-6))