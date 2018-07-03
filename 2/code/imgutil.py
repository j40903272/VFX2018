import cv2
import numpy as np
import transform
import math

def crop(img, edges, orig):
    """Crops an image so that it is a rectangle
    @npimg: an image to crop
    @return: the cropped image
    """
    # find the extents in the y direction
    maxX = orig[0, ...].max()
    maxY = orig[1, ...].max()
    x1 = 0 - edges[0, 0]
    x2 = maxX - edges[1, 0]
    y1 = 0 - edges[0, 1]
    y2 = -(maxY - edges[1, 1])

    # slice the image in y direction
    img = img[y1 + 1:img.shape[0] - y2 - 1]

    # find the extents in the x direction
    cropTop = np.argwhere(img[0, :, 3] != 0)
    cropBot = np.argwhere(img[-1, :, 3] != 0)
    minT = cropTop.min()
    maxT = cropTop.max()
    minB = cropBot.min()
    maxB = cropBot.max()

    # grab the correct extents
    xMin = max(minT, minB)
    xMax = min(maxT, maxB)

    # slice the image in x direction
    img = img[:, xMin:xMax]

    return img


def corners(homography, files):
    """Finds the corners of the images
    @homography: a list of homographies
    @return: an array of corners of the global window
    """
    # find the corners of all the images
    cornerL = []
    midCorners = None
    for i in range(len(files)):
        npimg = cv2.imread(files[i])

        # set up the corners in an array
        h, w = npimg.shape[0:2]
        corners = np.array([[0, w, w, 0], [0, 0, h, h]], dtype=float)
        corners = transform.homogeneous(corners)
        tform = homography[i]
        A = np.dot(tform, corners)
        A = transform.homogeneous(A)
        A = A.astype(int)
        cornerL.append(A)

        if i == len(files) / 2:
            midCorners = A

    # find the new corners of the image
    w1L = []
    w2L = []
    h1L = []
    h2L = []
    for i in range(len(cornerL)):
        w1L.append(np.min(cornerL[i][0, :]))
        w2L.append(np.max(cornerL[i][0, :]))
        h1L.append(np.min(cornerL[i][1, :]))
        h2L.append(np.max(cornerL[i][1, :]))
    w1 = min(w1L)
    w2 = max(w2L)
    h1 = min(h1L)
    h2 = max(h2L)

    # set up array to return
    ndarray = np.array([(w1, h1), (w2, h2)])

    return ndarray, midCorners


def ransac(data, tolerance=0.5, max_iterations=100, confidence=0.95):
    """Finds the best homography that maps the features between 2 images

    Input
    @data: an array of N x 4 point correspondences where each row 
           provides a point correspondence, (x1, y1, x2, y2)
    @tolerance: the error allowed for each data point to be an inlier
    @max_iterations: the maximum number of times to generate a random model

    Output
    model: the best model (a homography)
    inliers: the indices of the inliers in data 
    """
    # use a matrix to go along with the homography function
    data = np.matrix(data)

    iterations = 0
    best_model = None
    best_count = 0
    best_indices = None

    # if we reached the maximum iteration
    while iterations < max_iterations:

        # make two copies of the data
        temp_data = np.matrix(np.copy(data))
        temp_shuffle = np.copy(data)

        # shuffle the copied data and select 4 points
        np.random.shuffle(temp_shuffle)
        temp_shuffle = np.matrix(temp_shuffle)[0:4]

        # build a homography
        homo = homograph(temp_shuffle[:, 0:2], temp_shuffle[:, 2:])

        # grab the data for the appropriate image
        temp_data1 = temp_data[:, 0:2].transpose()
        temp_data1 = transform.homogeneous(temp_data1)
        temp_data2 = temp_data[:, 2:].transpose()

        # compute error for each point correspondence
        tform_pts = (homo * temp_data1)
        tform_pts = transform.homogeneous(tform_pts)[0:2, :]
        tform_pts = np.array(tform_pts)
        error = np.sqrt((np.array(tform_pts - temp_data2) ** 2).sum(0))

        inlier_count = (error < tolerance).sum()
        # if this homo is better than previous ones keep it
        if inlier_count > best_count:
            best_model = homo
            best_count = inlier_count
            best_indices = np.argwhere(error < tolerance)

            # recalculate max_iterations
            p = float(inlier_count) / data.shape[0]
            max_iterations = math.log(1 - confidence) / math.log(1 - (p ** 4))

        # increment iterations
        iterations += 1

    if best_model is None:
        raise ValueError("computed error never less than threshold")
    else:
        return best_model, best_indices

    
def homograph(p1, p2):
    """Finds the homography that maps points from p1 to p2
    @p1: a Nx2 array of positions that correspond to p2, N >= 4
    @p2: a Nx2 array of positions that correspond to p1, N >= 4
    @return: a 3x3 matrix that maps the points from p1 to p2
    p2=Hp1
    """
    # check if there is at least 4 points
    if p1.shape[0] < 4 or p2.shape[0] < 4:
        raise ValueError("p1 and p2 must have at least 4 row")

    # create matrix A
    A = np.zeros((p1.shape[0] * 2, 8), dtype=float)
    A = np.matrix(A, dtype=float)

    # fill A
    for i in range(0, A.shape[0]):
        # if i is event
        if i % 2 == 0:
            A[i, 0] = p1[i // 2, 0]
            A[i, 1] = p1[i // 2, 1]
            A[i, 2] = 1
            A[i, 6] = -p2[i // 2, 0] * p1[i // 2, 0]
            A[i, 7] = -p2[i // 2, 0] * p1[i // 2, 1]
        # if i is odd
        else:
            A[i, 3] = p1[i // 2, 0]
            A[i, 4] = p1[i // 2, 1]
            A[i, 5] = 1
            A[i, 6] = -p2[i // 2, 1] * p1[i // 2, 0]
            A[i, 7] = -p2[i // 2, 1] * p1[i // 2, 1]


    # create vector b
    b = p2.flatten()
    b = b.reshape(b.shape[1], 1)
    b = b.astype(float)


    # calculate homography Ax=b
    try:
        x = np.linalg.solve(A, b)
    except:
        x = np.linalg.lstsq(A, b)[0]

    # reshape x
    x = np.vstack((x, np.matrix(1)))
    x = x.reshape((3, 3))

    return x


