import concurrent.futures
import pandas as pd

nums = [1,2,3,4,5,6,7,8,9,10]

def f(x):
    return x * x
def main():
    # Make sure the map and function are working
    print([val for val in map(f, nums)])

    # Test to make sure concurrent map is working
    with concurrent.futures.ProcessPoolExecutor() as executor:
        print([val for val in executor.map(f, nums)])

if __name__ == '__main__':
    main()
