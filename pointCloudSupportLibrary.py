import math
from pyntcloud import PyntCloud
import pandas as pd
import numpy
# Filter points that are BEHIND, FAR AWAY and outside a certain view function

meanGroundPoint = 0

def filteruselesspoints(points, distance, lateralDistance, angle, fun=1, curvature = 10):
    newpc = []
    for point in points:
        # Filter behind point and far away point more than distance

        if point[1] <= 0 and point[1] ** 2 + point[0] ** 2 <= distance ** 2:
            # Linear function -> y = angle*x - lateraldistance
            # lateralDistance is the distance from the (0, 0) point
            # angle is the angle in degrees
            if fun == 1 and abs(point[1]) > (math.pi - (angle * math.pi) / 180) * abs(point[0]) - lateralDistance:
                newpc.append(point)

            # Logarithm function -> y = angle * log(10)( x - lateralDistance ) - curvature

            elif fun == 2 and abs(point[1]) > angle * math.log(abs(abs(point[0]) - lateralDistance), 10) - curvature:
                newpc.append(point)

            # Parabola function -> z + lateralDistance = distance*x^2 + y

            elif fun == 3 and point[2] + lateralDistance > distance * point[0] ** 2 + point[1]:
                newpc.append(point)

    return newpc



# Filter higher points that are higher than a function
# Z = A*Y + B

def filterHigherPoints(points, a,  b):

    newpc = []
    for point in points:
        if point[2] < a*point[1] + b:
            newpc.append(point)

    return newpc

# Ground Removal with RANSAC

def removeGround(pc, maxDistanceFromPlane, numberOfIteration):
    pcpanda = pd.DataFrame(pc)
    pcpanda.columns = ["x", "y", "z"]

    cloud = PyntCloud(pcpanda)
    is_floor = cloud.add_scalar_field("plane_fit", max_dist=maxDistanceFromPlane, max_iterations=numberOfIteration)
    pcpanda = pd.DataFrame(cloud.points)
    groundPoints = pcpanda[pcpanda.is_plane == 1]
    del groundPoints['is_plane']
    points = groundPoints.values
    meanGroundPoint = numpy.mean(points[:, 2])
    print(meanGroundPoint)
    pointCloudWithoutGround = pcpanda[pcpanda.is_plane != 1]
    del pointCloudWithoutGround['is_plane']

    return pointCloudWithoutGround.values
