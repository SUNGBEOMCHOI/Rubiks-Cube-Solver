import time
import multiprocessing

start_time = time.time()
def count(name):
    for i in range(1, 1000):
        print(name, " : ", i)
        pass
    
num_list = ['p1', 'p2', 'p3', 'p4']

for num in num_list:
    count(num)

print(time.time()-start_time)



# if __name__ == "__main__":
#     pool = multiprocessing.Pool(processes=8)
#     pool.map(count, num_list)
#     pool.close()
#     pool.join()
#     print(time.time()-start_time)