import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from scipy.spatial.distance import directed_hausdorff

# 5 random points for the first set of points
pts_1 = np.random.uniform(-1, 1, (5, 3))
hull_1 = ConvexHull(pts_1)

# 5 random points for the second set of points
pts_2 = np.random.uniform(-1, 1, (5, 3))
hull_2 = ConvexHull(pts_2)

# Compute the Hausdorff distance between the two convex hulls
hull_1_points = pts_1[hull_1.vertices]
hull_2_points = pts_2[hull_2.vertices]

# Compute directed Hausdorff distances
d_1_to_2 = directed_hausdorff(hull_1_points, pts_2)[0]
d_2_to_1 = directed_hausdorff(pts_2, hull_1_points)[0]

hausdorff_distance = max(d_1_to_2, d_2_to_1)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot points for the first set
ax.plot(pts_1.T[0], pts_1.T[1], pts_1.T[2], "bo", label="Points 1")

# Plot points for the second set
ax.plot(pts_2.T[0], pts_2.T[1], pts_2.T[2], "go", label="Points 2")

# Plot the convex hulls of both point sets
for s in hull_1.simplices:
    s = np.append(s, s[0])  # Close the loop of the face
    ax.plot(pts_1[s, 0], pts_1[s, 1], pts_1[s, 2], "b-", alpha=0.6)

for s in hull_2.simplices:
    s = np.append(s, s[0])  # Close the loop of the face
    ax.plot(pts_2[s, 0], pts_2[s, 1], pts_2[s, 2], "g-", alpha=0.6)

# Plot centroids for each set of points
centroid_1 = np.mean(pts_1, axis=0)
centroid_2 = np.mean(pts_2, axis=0)
ax.plot([centroid_1[0]], [centroid_1[1]], [centroid_1[2]], "bo")
ax.plot([centroid_2[0]], [centroid_2[1]], [centroid_2[2]], "go")

# Labels and legend
for i in ["x", "y", "z"]:
    eval("ax.set_{:s}label('{:s}')".format(i, i))

# Display the Hausdorff distance on the plot
ax.text2D(0.05,
          0.95,
          f"Hausdorff distance: {hausdorff_distance:.3f}",
          transform=ax.transAxes)
ax.legend()

plt.show()
