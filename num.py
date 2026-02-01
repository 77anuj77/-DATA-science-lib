"""numpy is a powerful numerical computing library in Python that provides support for
large, multi-dimensional arrays and matrices, along with a collection of mathematical
functions. it is built in c language and is more storage and time efficient than list in Python."""

import numpy as np

# Creating a numpy array from a list
data = [1, 2, 3, 4, 5]
arr = np.array(data)
print(arr)
#we can use list,tuple, dictionary as well
data={1:10,2:20,3:30}
arr=np.array(data.keys)
arr=np.array(data.values)
arr=np.array(data.items)
# the dimension in the numpy is determined by the number of brackets

data = [[]]
arr = np.array(data)
print(arr.ndim)

# cration of arraay using arrange funtion

arr = np.arange(0, 10, 2)
arr1 = np.arange(0, 1, 0.5) 
arr1 # but for evenly increment we use linspace
arr2 = np.linspace(0, 1, 5)
arr2
# logshape finction is for printing logarethmic space logspace(start, ending-> powers, 3->numbers of elenments)
arr3 = np.logspace(0, 2, 3)

# np.zeros and np.ones function is used to create array of zeros and ones
arr4 = np.zeros((2, 3))  # 3 rows and 2 columns
print(arr4)
arr5 = np.ones((3, 2))  # instead of '()' we can use '[]' also
print(arr5)

# creating your own array using custom values using full function
arr = np.full((4), 7)
arr
arr = np.full((3, 4), 5)
arr

# creating an array eithout setting values usong empty function
arr = np.empty((2, 3))
arr  # thia method is for oveerwriting the exixting values in the array
arr = np.empty((2, 3), dtype=int)
arr  # specifying the datatype of the array

# random funtion
arr = np.random.rand(3, 2)  # uniform distribution between 0 and 1
arr
arr = np.random.randn(3, 2)  # normal distribution with mean 0 and variance 1
arr
arr = np.random.randint(0, 10, (3, 2))  # random integers between 0 and 10
arr

# numpy datatypes andtype casting
arr = np.array([1, 2, 3])
arr.dtpype  # checking datatype of the array
arr = np.array([1.5, 2.5, 3.5])
arr.dtype
arr = np.array([1, 2, 3], dtype=float)  # specifying datatype while creating array
arr.dtype
arr = arr.astype(int)  # type casting owerwriting the existing datatype
arr.dtype
# typecasting errors-> WE CANNOT CONVERT STRING TO INT or float

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr.size  # total number of elements in the array
arr.shape  # shape of the array
arr.ndim  # number of dimensions of the array
arr.dtype  # datatype of the array
arr.itemsize  # size of each element in bytes
arr.nbytes  # total size of the array in bytes

# reshaping of the array
arr = np.array([1, 2, 3, 4, 5, 6])
arr.reshape(2, 3)
arr.reshape(3, 2)

# ravel funtion to convert multi dimensional array to 1D array
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr
ravel = arr.ravel()
ravel
ravel[0] = 10
ravel
arr
"""ravel function creates a view of the original array, so any changes made
to the raveled array will affect the original array."""

# flatten function to convert multi dimensional array to 1D array
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr
flat = arr.flatten()
flat
flat[0] = 10
flat
arr
"""flatten function creates a copy of the original array, so any changes made
to the flattened array will not affect the original array."""

angles = np.array([0, np.pi / 2, np.pi])
sin_angle = np.sin(angles)
sin_angle

# indexing and slicing in 1D array
arr = np.arange(10, 110, 10)
arr
arr[0]  # first element
arr[5]  # sixth element
arr[-1]  # last element
arr[2:5]  # slicing from index 2 to 4
arr[::2]  # slicing with step size 2
arr[::-1]  # reversing the array

# 2D array indexing and slicing
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr
print(arr[0, 1, 2] ) # accessing element at 0th row, 1st column, 2nd depth
arr
arr[1, 0, 1]  # accessing elements at 1st row, 0th column, 1st depth
arr[0, :, :]  # accessing all columns and depths of 0th row
arr[:, 1, :]  # accessing all rows and depths of 1st column
arr[:, :, 2]  # accessing all rows and columns of 2nd depth
arr[0, 1, :]  # accessing all depths of 0th row and 1st column
arr[:, 0, 1]  # accessing all rows and 1st depth of 0th column
arr[1, :, 0]  # accessing all columns and 0th depth of 1st row
arr[0, 3:0, 2]  # slicing in 3D array


"""np.take is a built in function that allows slicing elements from 
array and indexing based on the specified indices."""
arr = np.array([1, 2, 3, 4, 5])
index = [0, 2]
slicing = np.take(arr, index)
slicing


# concatination ,vstack and hstack funtions
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
conc = np.concatenate((arr1, arr2))
conc
hstack = np.hstack((arr1, arr2))
hstack
vstack = np.vstack((arr1, arr2))
vstack

# splitting of arrays using np.split, np.hsplit, np.vsplit functions
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
split = np.split(arr, 2)
split
hsplit = np.hsplit(arr, 2)
hsplit
vsplit = np.vsplit(arr, 2)
vsplit

# repeating and tiling of arrays using np.repeat and np.tile functions
arr = np.array([1, 2, 3])
rep = np.repeat(arr, 3)#repeat every element 3 times 
rep
tile = np.tile(arr, 3)#repeat a list 3 times in a row
tile

# aggrigate function in numpy
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr
arr.sum()  # sum of all elements
arr.sum(axis=0)  # sum of columns
arr
arr.sum(axis=1)  # sum of rows
arr
arr.mean()  # mean of all elements
arr.mean(axis=0)  # mean of columns
arr
arr.mean(axis=1)  # mean of rows
arr
arr.std()  # standard deviation of all elements
arr.std(axis=0)  # standard deviation of columns
arr
arr.std(axis=1)  # standard deviation of rows
arr
arr.min()  # minimum of all elements
arr.min(axis=0)  # minimum of columns
arr.min(axis=1)  # minimum of rows
arr.max()  # maximum of all elements
arr.max(axis=0)  # maximum of columns
arr.max(axis=1)  # maximum of rows
arr.argmin()  # index of minimum element
arr.argmax()  # index of maximum element
arr.cumsum()  # cumulative sum of all elements
arr.cumsum(axis=0)  # cumulative sum of columns
arr
arr.cumsum(axis=1)  # cumulative sum of rows
arr
arr.cumprod()  # cumulative product of all elements
arr.cumprod(axis=0)  # cumulative product of columns
arr
arr.cumprod(axis=1)  # cumulative product of rows
arr
arr.prod()  # product of all elements
arr.prod(axis=0)  # product of columns
arr
arr.prod(axis=1)  # product of rows
arr
arr.var()  # variance of all elements
arr.var(axis=0)  # variance of columns
arr
arr.var(axis=1)  # variance of rows
arr
arr.ptp()  # peak to peak (max-min) of all elements
arr.ptp(axis=0)  # peak to peak of columns
arr
arr.ptp(axis=1)  # peak to peak of rows
arr
arr.flatten().sum()  # sum using flatten
arr.ravel().sum()  # sum using ravel
arr.flatten().mean()  # mean using flatten
arr.ravel().mean()  # mean using ravel
arr.flatten().std()  # standard deviation using flatten
arr.ravel().std()  # standard deviation using ravel

"""np.where function is used to return the indices of elements
that satisfy a given condition."""
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
indices = np.where(arr % 2 == 0)
indices

"""NaN (Not a Number) handling through nan funtion in numpy"""
arr= np.array([1, 2, np.nan, 4, 5, np.nan])
arr
np.isnan(arr)  # checking for NaN values
np.nanmean(arr)  # mean ignoring NaN values
np.nanstd(arr)  # standard deviation ignoring NaN values
np.nansum(arr)  # sum ignoring NaN values
np.nanmin(arr)  # minimum ignoring NaN values
np.nanmax(arr)  # maximum ignoring NaN values
np.nanargmin(arr)  # index of minimum ignoring NaN values
np.nanargmax(arr)  # index of maximum ignoring NaN values
np.nanmedian(arr)  # median ignoring NaN values
np.nanvar(arr)  # variance ignoring NaN values
np.nanprod(arr)  # product ignoring NaN values
np.nanpercentile(arr, 50)  # 50th percentile ignoring NaN values
np.nanquantile(arr, 0.5)  # 50th quantile ignoring NaN values
np.nancumsum(arr)  # cumulative sum ignoring NaN values
np.nancumprod(arr)  # cumulative product ignoring NaN values
np.nan_to_num(arr)  # replace NaN with 0
np.nan_to_num(arr, nan=100)  # replace NaN with 100
np.isnan(arr) # checking for NaN values again

arr=np.array([1,2,np.inf,-3, -np.inf])
np.isinf(arr)  # checking for infinite values
np.nan_to_num(arr, posinf=100, neginf=-100)  # replace inf with 100 and -inf with -100
arr
arr.astype(int)