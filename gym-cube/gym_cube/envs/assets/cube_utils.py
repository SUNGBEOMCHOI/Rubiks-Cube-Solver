from pycuber import *

def isSolved_(sim_cube):
    solved = True
    if sim_cube.D == [[Square(sim_cube["D"].colour)] * 3] * 3:
        for face in "LFRB":
            if sim_cube.get_face(face)[:] != [[Square(sim_cube[face].colour)] * 3] * 3:
                solved = False
    else:
        solved = False
    return solved