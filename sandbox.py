import opensim as osim


model = osim.Model('unscaled_generic.osim')
state = model.initSystem()
matter = model.getMatterSubsystem()


# matrix = osim.Matrix()
# points = [0, 1, 2]
# vecs = [osim.Vec3(1,0,0), osim.Vec3(0,1,0), osim.Vec3(0,0,1)]
# matter.calcStationJacobian(state, points, vecs, matrix)

import pdb; pdb.set_trace()