import numpy as np
'''

            0 1 2
            3 4 5
            6 7 8

36 37 38    18 19 20    9 10 11     45 46 47
39 40 41    21 22 23    12 13 14    48 49 50
42 43 44    24 25 26    15 16 17    51 52 53

            27 28 29
            30 31 32
            33 34 35

face colors:
    ┌──┐
    │ 0│
 ┌──┼──┼──┬──┐
 │ 4│ 2│ 1│ 5│
 └──┼──┼──┴──┘
    │ 3│
    └──┘

TODO: moveDefs -> pieceDefs -> hashOP -> pieceInds
[   0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53], \

[   0,  1,  2,  3,  4,  5,  6,  7,  8,  \
    9,  10, 11, 12, 13, 14, 15, 16, 17, \
    18, 19, 20, 21, 22, 23, 24, 25, 26, \
    27, 28, 29, 30, 31, 32, 33, 34, 35, \
    36, 37, 38, 39, 40, 41, 42, 43, 44, \
    45, 46, 47, 48, 49, 50, 51, 52, 53],\


[
    4 5 5 2 0 0 2 0 0 
    2 2 0 1 1 1 1 1 1 
    5 4 4 3 2 0 3 2 0 
    3 3 2 5 3 2 1 3 2 
    3 5 5 4 4 4 4 4 4 
    1 1 0 3 5 0 3 5 5
    ]

corner가 가질 수 없는 경우의 수가 나온다 -> pieceInds에서 max size를 초과하는 index발생

1. moveDefs 문제
2. hashOp 문제
3. pieceInds 문제
'''
moveInds = { \
  "U": 0, "U'": 1, "F": 2, "F'": 3, "R": 4, "R'": 5, 
  "D": 6, "D'": 7, "B": 8, "B'": 9, "L": 10, "L'": 11
}

moveDefs = np.array([ \
# U
[    6,  3,  0,  7,  4,  1,  8,  5,  2, \
    45, 46, 47, 12, 13, 14, 15, 16, 17, \
     9, 10, 11, 21, 22, 23, 24, 25, 26, \
    27, 28, 29, 30, 31, 32, 33, 34, 35, \
    18, 19, 20, 39, 40, 41, 42, 43, 44, \
    36, 37, 38, 48, 49, 50, 51, 52, 53],\
# U'
[   2,  5,  8,  1,  4,  7,  0,  3,  6,  \
    18, 19, 20, 12, 13, 14, 15, 16, 17, \
    36, 37, 38, 21, 22, 23, 24, 25, 26, \
    27, 28, 29, 30, 31, 32, 33, 34, 35, \
    45, 46, 47, 39, 40, 41, 42, 43, 44, \
     9, 10, 11, 48, 49, 50, 51, 52, 53],\
# F
[   0,  1,  2,  3,  4,  5, 44,  41, 38, \
    6,  10, 11, 7, 13, 14, 8, 16, 17,   \
    24, 21, 18, 25, 22, 19, 26, 23, 20, \
    15, 12, 9, 30, 31, 32, 33, 34, 35,  \
    36, 37, 27, 39, 40, 28, 42, 43, 29, \
    45, 46, 47, 48, 49, 50, 51, 52, 53],\
# F'   
[   0,  1,  2,  3,  4,  5,  9,  12,  15,\
    29,  10, 11, 28, 13, 14, 27, 16, 17,\
    20, 23, 26, 19, 22, 25, 18, 21, 24, \
    38, 41, 44, 30, 31, 32, 33, 34, 35, \
    36, 37, 8, 39, 40, 7, 42, 43, 6,    \
    45, 46, 47, 48, 49, 50, 51, 52, 53],\
# R
[   0,  1,  20,  3,  4, 23,  6, 7,  26, \
    15,  12, 9, 16, 13, 10, 17, 14, 11, \
    18, 19, 29, 21, 22, 32, 24, 25, 35, \
    27, 28, 51, 30, 31, 48, 33, 34, 45, \
    36, 37, 38, 39, 40, 41, 42, 43, 44, \
    8, 46, 47, 5, 49, 50, 2, 52, 53],   \
# R'
[   0,  1,  51, 3,  4,  48,  6, 7,  45, \
    11,  14, 17, 10, 13, 16, 9, 12, 15, \
    18, 19, 2, 21, 22, 5, 24, 25, 8,    \
    27, 28, 20, 30, 31, 23, 33, 34, 26, \
    36, 37, 38, 39, 40, 41, 42, 43, 44, \
    35, 46, 47, 32, 49, 50, 29, 52, 53],\
# D
# 5th: 51, 52, 53 -> 53, 52, 51
[   0,  1,  2,  3,  4,  5,  6,  7,  8,  \
    9,  10, 11, 12, 13, 14, 24, 25, 26, \
    18, 19, 20, 21, 22, 23, 42, 43, 44, \
    33, 30, 27, 34, 31, 28, 35, 32, 29, \
    36, 37, 38, 39, 40, 41, 51, 52, 53, \
    45, 46, 47, 48, 49, 50, 15, 16, 17],\
# D'
# 6th: 42, 43, 44 -> 44, 43, 42
[   0,  1,  2,  3,  4,  5,  6,  7,  8,  \
    9,  10, 11, 12, 13, 14, 51, 52, 53, \
    18, 19, 20, 21, 22, 23, 15, 16, 17, \
    29, 32, 35, 28, 31, 34, 27, 30, 33, \
    36, 37, 38, 39, 40, 41, 24, 25, 26, \
    45, 46, 47, 48, 49, 50, 42, 43, 44],\

# B
[   11,  14,  17,  3,  4,  5,  6, 7, 8, \
    9,  10, 35, 12, 13, 34, 15, 16, 33, \
    18, 19, 20, 21, 22, 23, 24, 25, 26, \
    27, 28, 29, 30, 31, 32, 36, 39, 42, \
    2, 37, 38, 1, 40, 41, 0, 43, 44,    \
    51, 48, 45, 52, 49, 46, 53, 50, 47],\
# B'
# 5th: 25 -> 35
[   42,  39,  36,  3,  4,  5,  6, 7, 8, \
    9,  10, 0, 12, 13, 1, 15, 16, 2,    \
    18, 19, 20, 21, 22, 23, 24, 25, 26, \
    27, 28, 29, 30, 31, 32, 17, 14, 11, \
    33, 37, 38, 34, 40, 41, 35, 43, 44, \
    47, 50, 53, 46, 49, 52, 45, 48, 51],\
# L
# L face 수정완료
[   53,  1,  2,  50,  4,  5, 47, 7, 8,  \
    9,  10, 11, 12, 13, 14, 15, 16, 17, \
    0, 19, 20, 3, 22, 23, 6, 25, 26,    \
    18, 28, 29, 21, 31, 32, 24, 34, 35, \
    42, 39, 36, 43, 40, 37, 44, 41, 38, \
    45, 46, 33, 48, 49, 30, 51, 52, 27],\
# L'
# 3th: 53 -> 27
[   18,  1,  2,  21,  4, 5,  24, 7, 8,  \
    9,  10, 11, 12, 13, 14, 15, 16, 17, \
    27, 19, 20, 30, 22, 23, 33, 25, 26, \
    53, 28, 29, 50, 31, 32, 47, 34, 35, \
    38, 41, 44, 37, 40, 43, 36, 39, 42, \
    45, 46, 6, 48, 49, 3, 51, 52, 0],   \

])# [12, 54]

corner_pieceDefs = np.array([ \
  [ 0, 47, 36], \
  [ 6, 38, 18], \
  [ 8, 20,  9], \
  [ 2, 11, 45], \
  [33, 42, 53], \
  [27, 24, 44], \
  [29, 26, 15], \
  [35, 51, 17], \
]) # [8, 3]

edge_pieceDefs = np.array([\
  [  1, 46], \
  [  3, 37], \
  [  7, 19], \
  [  5, 10], \
  [ 34, 52], \
  [ 30, 43], \
  [ 28, 25], \
  [ 32, 16], \
  [ 21, 41], \
  [ 23, 12], \
  [ 48, 14], \
  [ 50, 39], \
    ]) # [12, 2]

# how to choose hash number?
corner_hashOP = np.array([1, 2, 10]) # 3d
edge_hashOP = np.array([1, 10]) # 2d

# [max hash number, [cubelet position, face position]] z y x
corner_pieceInds = np.zeros([62, 2], dtype=np.int)
corner_pieceInds[50] = [0, 0]; corner_pieceInds[54] = [0, 1]; corner_pieceInds[13] = [0, 2]
corner_pieceInds[28] = [1, 0]; corner_pieceInds[ 8] = [1, 1]; corner_pieceInds[42] = [1, 2]
corner_pieceInds[14] = [2, 0]; corner_pieceInds[ 5] = [2, 1]; corner_pieceInds[12] = [2, 2]
corner_pieceInds[52] = [3, 0]; corner_pieceInds[11] = [3, 1]; corner_pieceInds[15] = [3, 2]

corner_pieceInds[61] = [4, 0]; corner_pieceInds[44] = [4, 1]; corner_pieceInds[51] = [4, 2]
corner_pieceInds[47] = [5, 0]; corner_pieceInds[30] = [5, 1]; corner_pieceInds[40] = [5, 2] # 60 -> 40
corner_pieceInds[17] = [6, 0]; corner_pieceInds[35] = [6, 1]; corner_pieceInds[18] = [6, 2]
corner_pieceInds[23] = [7, 0]; corner_pieceInds[56] = [7, 1]; corner_pieceInds[21] = [7, 2]

edge_pieceInds = np.zeros([55, 2], dtype=np.int)
# 40과 43이 중복이다 왜??
edge_pieceInds[50] = [0, 0]; edge_pieceInds[ 5] = [0, 1]
edge_pieceInds[40] = [1, 0]; edge_pieceInds[ 4] = [1, 1]
edge_pieceInds[20] = [2, 0]; edge_pieceInds[ 2] = [2, 1]
edge_pieceInds[10] = [3, 0]; edge_pieceInds[ 1] = [3, 1] # 13 -> 10

edge_pieceInds[53] = [4, 0]; edge_pieceInds[35] = [4, 1]
edge_pieceInds[43] = [5, 0]; edge_pieceInds[34] = [5, 1]
edge_pieceInds[23] = [6, 0]; edge_pieceInds[32] = [6, 1]
edge_pieceInds[13] = [7, 0]; edge_pieceInds[31] = [7, 1]

edge_pieceInds[42] = [8, 0]; edge_pieceInds[24] = [8, 1]
edge_pieceInds[12] = [9, 0]; edge_pieceInds[21] = [9, 1]

edge_pieceInds[15] = [10, 0]; edge_pieceInds[51] = [10, 1]
edge_pieceInds[45] = [11, 0]; edge_pieceInds[54] = [11, 1]



# init State: [54, ]
# corner와 edge 각각의 OP를 만들자
# getOP
# -> np.dot(s[pieceDefs], hashOP)) : get sticker of corner or edge -> multiply hash
# -> return hash number
# pieceInds[hash number] = [cubelet position, face position]
# pieceInds for corner: [max hash number, [8, 3]]
# pieceInds for edge: [max hash number, [12, 2]]

def initState_3():
    return np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0,\
        1, 1, 1, 1, 1, 1, 1, 1, 1,\
        2, 2, 2, 2, 2, 2, 2, 2, 2,\
        3, 3, 3, 3, 3, 3, 3, 3, 3,\
        4, 4, 4, 4, 4, 4, 4, 4, 4,\
        5, 5, 5, 5, 5, 5, 5, 5, 5])
    
def doMove_3(s, move):
    move = moveInds[move]
    return s[moveDefs[move]]

def getOP_3(s):
    try:
        corner_op = corner_pieceInds[np.dot(s[corner_pieceDefs], corner_hashOP)]
    except:
        print("Corner error: ", s[corner_pieceDefs])
    try:
        edge_op = edge_pieceInds[np.dot(s[edge_pieceDefs], edge_hashOP)]
    except:
        print("Edge error: ", s[edge_pieceDefs])
    return np.concatenate((corner_op, edge_op))

def isSolved_3(s):
  for i in range(6):
    if not (s[9 * i:9 * i + 9] == s[9 * i]).all():
      return False
  return True

def pos_to_state_3(pos):
    # pos: 20, 2 -> [cube name, [cubelet position, face position]]
    # state: 20, 24
    state = np.zeros([20, 24], dtype=np.int)
    for position, pos_element in enumerate(pos):
        if position < 8:
            state_value = pos_element[0] * 3 + pos_element[1]
        else:
            state_value = pos_element[0] * 2 + pos_element[1]
    
        state[position][state_value] = 1.0
    return state
