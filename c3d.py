import os
import numpy as np
import opensim as osim
import ezc3d

class C3D:
    def __init__(self, filepath):
        self.filepath = filepath
        self.c3d = ezc3d.c3d(filepath)

        # This is a Y-Z space-fixed rotation needed to convert data collected from Theia
        # to OpenSim's ground reference frame convention (X forward, Y up, Z right).
        data_rotation = osim.Rotation()
        data_rotation.setRotationFromTwoAnglesTwoAxes(1, # space-fixed
                -0.5*np.pi, osim.CoordinateAxis(1), # Y rotation
                -0.5*np.pi, osim.CoordinateAxis(2)) # Z rotation
        self.data_rotation = data_rotation

        # This is an additional body-fixed rotation that effectively swaps the axes of
        # the rotations collected from Theia to match OpenSim's ground reference frame
        # convention (X forward, Y up, Z right), which is the convention used by the
        # matching Frame elements in the generic model.
        frame_rotation = osim.Rotation()
        frame_rotation.setRotationToBodyFixedXY(osim.Vec2(0.5*np.pi))
        self.frame_rotation = frame_rotation

    def get_data(self, parameter):
        return self.c3d.data[parameter]

    def get_data_labels(self, parameter):
        return self.c3d.parameters[parameter]['LABELS']['value']

    def get_data_rate(self, parameter):
        return self.c3d.parameters[parameter]['RATE']['value'][0]

    def get_time_vector(self, rate, num_frames):
        return np.array([i/rate for i in range(num_frames)])

    def get_frame_positions_table(self):
        data = self.get_data('rotations')
        num_frames = data.shape[3]
        labels = [label.replace('_4X4', '') for label in self.get_data_labels('ROTATION')]
        rate = self.get_data_rate('ROTATION')
        times = self.get_time_vector(rate, num_frames)

        table = osim.TimeSeriesTableVec3()
        for iframe in range(num_frames):
            row = osim.RowVectorVec3(len(labels), osim.Vec3(0))
            for ilabel, label in enumerate(labels):
                position = data[:, 3, ilabel, iframe] / 1000.0  # mm to m
                row[ilabel] = osim.Vec3(position[0], position[1], position[2])
                row[ilabel] = self.rotation.multiply(row[ilabel])

            table.appendRow(times[iframe], row)

        table.setColumnLabels(labels)
        table.addTableMetaDataString("Units", "m")
        table.addTableMetaDataString("DataRate", str(rate))

        return table

    def get_frame_rotations_table(self):
        data = self.get_data('rotations')
        num_frames = data.shape[3]
        labels = [label.replace('_4X4', '') for label in self.get_data_labels('ROTATION')]
        rate = self.get_data_rate('ROTATION')
        times = self.get_time_vector(rate, num_frames)

        table = osim.TimeSeriesTableQuaternion()
        for iframe in range(num_frames):
            row = osim.RowVectorQuaternion(len(labels), osim.Quaternion())
            for ilabel, label in enumerate(labels):
                rot = data[:3, :3, ilabel, iframe]
                mat33 = osim.Mat33(rot[0,0], rot[0,1], rot[0,2],
                                   rot[1,0], rot[1,1], rot[1,2],
                                   rot[2,0], rot[2,1], rot[2,2])
                rotation = self.rotation.multiply(osim.Rotation(mat33))
                rotation = rotation.multiply(self.rotation2)

                # Store as a quaternion.
                new_quat = rotation.convertRotationToQuaternion()
                upd_quat = row.updElt(0, ilabel)
                upd_quat.set(0, new_quat.get(0))
                upd_quat.set(1, new_quat.get(1))
                upd_quat.set(2, new_quat.get(2))
                upd_quat.set(3, new_quat.get(3))

            table.appendRow(times[iframe], row)

        table.setColumnLabels(labels)
        table.addTableMetaDataString("Units", "m")
        table.addTableMetaDataString("DataRate", str(rate))

        return table


c3d = C3D(os.path.join('data', 'acl', 'theia', 'jump_1', 'pose_0.c3d'))
positions = c3d.get_frame_positions_table()
rotations = c3d.get_frame_rotations_table()

trc = osim.TRCFileAdapter()
trc.write(positions, 'positions.trc')

sto = osim.STOFileAdapterQuaternion()
sto.write(rotations, 'rotations.sto')


# print(dir(c))
# print(c.parameters.ROTATION.LABELS)
# import pdb; pdb.set_trace()

