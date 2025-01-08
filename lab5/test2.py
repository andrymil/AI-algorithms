# Create an array
array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Group every other element with each other
grouped_array = [(array[i], array[i + 1]) for i in range(0, len(array), 2)]

print(grouped_array)