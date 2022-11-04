import pycuber as pc
import time
import numpy as np

rot_cnt = 60000
# Create a Cube object
mycube = pc.Cube()

# Create the list of Actions
action = ["R", "R'", "L", "L'", "U", "U'", "D", "D'", "F", "F'", "B", "B'"]
# Do something at the cube.
# mycube("R U R' U'")

print("="*80)

start_sameone = time.time()
for _ in range(rot_cnt):
    mycube("R")
end_sameone = time.time()
print(f"""Rotate {rot_cnt} times with same action "R":\t\t{end_sameone-start_sameone}""")
print()

start_changing = time.time()
i = 0
for _ in range(rot_cnt):
    mycube(action[i])
    i += 1
    if i == 12:
        i = 0
end_changing = time.time()
print(f"""Rotate {rot_cnt} times with action changing:\t\t{end_changing-start_changing}""")
print()

start_make_concat = time.time()
action_sequence = " ".join(action*(int(rot_cnt/len(action))))
end_make_concat = time.time()
start_changing_concat = time.time()
mycube(action_sequence)
end_changing_concat = time.time()
print(f"""Make sequence string with rotate {rot_cnt} times:\t\t{end_make_concat-start_make_concat}""")
print(f"""Rotate {rot_cnt} times with one string:\t\t\t{end_changing_concat-start_changing_concat}""")
print()

possible_numbers = np.array([i for i in range(12)])

start_random = time.time()
for _ in range(rot_cnt):
    mycube(action[np.random.choice(possible_numbers, size=1)[0]])
end_random = time.time()
print(f"""Rotate {rot_cnt} times with picking random at each step:\t{end_random-start_random}""")
print()

start_make_array = time.time()
sequence = np.random.choice(possible_numbers, size=rot_cnt)
end_make_array = time.time()
print(f"""Make random index sequence array:\t\t\t{end_make_array-start_make_array}""")

start_random_seq = time.time()
for i in range(rot_cnt):
    mycube(action[sequence[i]])
end_random_seq = time.time()
print(f"""Rotate {rot_cnt} times with predefined index sequences:\t{end_random_seq-start_random_seq}""")
print()

start_random_make_string = time.time()
sequence_string = " ".join([action[i] for i in sequence])
end_random_make_string = time.time()
print(f"""Make random action sequence array:\t\t\t{end_random_make_string-start_random_make_string}""")

start_random_changing = time.time()
mycube(sequence_string)
end_random_changing = time.time()
print(f"""Rotate {rot_cnt} times with predefined action string:\t{end_random_changing-start_random_changing}""")
print("="*80)
