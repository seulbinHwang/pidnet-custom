import scipy.io

# Load the color150.mat file
mat = scipy.io.loadmat('color150.mat')

# Extract the color array
colors = mat['colors']
print("colors:", colors)
print("colors.shape:", colors.shape)
# Now colors[i] gives the RGB color for class i
