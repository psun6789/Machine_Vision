'''
Name: Peter Sunny Shanthveer Markappa
Student Number: R00208303
Subject: Machine Vision
Assignment: 02
Date of Submission: 29th April, 2022
Document Status: Final

'''



# importing the libraries
import numpy as np
import cv2, os, glob, random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




def task_1_AB():
    '''
        --------- Checker board task------------------
        a) loading the zip file
        b) extraction and display of checker board corners to subpixel
        c) camera calibration
        d) principal length, aspect ratio, principal point
    '''
    image_points = [] ## 2 dimension points in image plane.
    obj_points = [] ## 3 dimension point in real world space

    # reference ---- https: // programtalk.com / python - examples / cv2.TERM_CRITERIA_EPS /
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    row = 5
    col = 7
    points = 3
    object_points = np.zeros((row * col, points), dtype='float32')
    meshgrid = np.mgrid[0:col, 0:row].T.reshape(-1, 2)
    object_points[:, :(points - 1)] = meshgrid

    '''
        # reference
        # https://stackoverflow.com/questions/66225558/cv2-findchessboardcorners-fails-to-find-corners
        # https://docs.opencv.org/4.x/dd/d92/tutorial_corner_subpixels.html
        # https://opencv24-python-tutorials.readthedocs.io/en/stable/py_tutorials/py_calib3d/py_calibration/py_calibration.html#:~:text=calibrateCamera().,%2C%20rvecs%2C%20tvecs%20%3D%20cv2.
    '''
    imdir = 'Assignment_MV_02_calibration/'
    winSize = (11, 11)
    zeroZone = (-1, -1)
    for name in glob.glob(imdir + '*.PNG'):
        image = cv2.imread(name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Gray Image", gray)
        cv2.waitKey(0)

        ret, corners = cv2.findChessboardCorners(gray, (col, row),
                                            flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                            cv2.CALIB_CB_FAST_CHECK +
                                            cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            corner = cv2.cornerSubPix(gray, corners, winSize, zeroZone, criteria=criteria)
            image_points.append(corner)
            obj_points.append(object_points)

            img1 = cv2.drawChessboardCorners(image, (col, row), corner, ret)

            # # Display the Corner Points
            cv2.imshow("Checker Board Corner points", img1)
            cv2.waitKey(0)
        else:
            print("*** No Checker Board Corner Found ***")

    # Camera Calibrate
    ret, matrix, distortion, rotation_vectors, translation_vectors = cv2.calibrateCamera(obj_points, image_points, gray.shape[::-1], None, None)

    # https://stackoverflow.com/questions/50421178/camera-projection-matrix-principal-point
    print("Camera Matrix \n", matrix)
    print("Camera distortion", distortion)
    print("Camera rotation_vectors", rotation_vectors)
    print("Camera translation_vectors", translation_vectors)
    m0 = matrix[0, 0]
    m1 = matrix[0, 2]
    m2 = matrix[1, 1]
    m3 = matrix[1, 2]
    principal_point = (m1, m3)
    principal_length = m0
    aspect_ratio = m2 / m0
    print("Principal Point \n", principal_point)
    print("principal length  \n", principal_length)
    print("aspect ratio  \n", aspect_ratio)

    return matrix, image_points, object_points

#---------------------------- END of TASK 1 AB ------------------------------------------
#----------------------------------------------------------------------------------------


#---------------------------- START of TASK 1 CD ------------------------------------------

def task_1_BC():

    # reference ---- https: // programtalk.com / python - examples / cv2.TERM_CRITERIA_EPS /
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    winSize = (11, 11)
    zeroZone = (-1, -1)

    # Reading the mp4 video for tracking the good features
    video_capture = cv2.VideoCapture("Assignment_MV_02_video.mp4")

    '''  ----   Reference from LAB-06 ----------'''
    # Extracting the initial frame of the captured video
    ret, first_frame = video_capture.read()

    if ret == True:
        print("--------- Task 1 BC ------------")

        cv2.imshow("First_Frame", first_frame)
        cv2.waitKey(0)

        # converting frame from RGB to Gray Scale
        gray_scale = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        # from the first frame extracting the interesting points
        features = cv2.goodFeaturesToTrack(gray_scale, 200, 0.3, 7)
        print("\n length of the features \n", len(features))
        print("\n features \n", features[0:len(features), :, :])
        print("good features")

        # from first frame we are extracting the interest points with sub-pixel accuracy
        subpixel_features = cv2.cornerSubPix(gray_scale, features, winSize, zeroZone, criteria=criteria)

        print("\n length of the sub-pixel features \n", len(subpixel_features))
        print("\n sub pixel features \n", features[0:len(features), :, :])



        index = np.arange(len(subpixel_features))
        tracks = {}
        sub_pixel_length = len(subpixel_features)

        for i in range(sub_pixel_length):
            tracks[index[i]] = {0: subpixel_features[i]}

        print("\ntracks: \n", tracks)

        frame = 0
        while ret:
            ret, img = video_capture.read()

            if not ret:
                break

            frame += 1
            old_img = gray_scale
            gray_scale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            if len(subpixel_features) > 0:
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_img, gray_scale, subpixel_features, None)

                # visualise points
                for i in range(len(st)):
                    if st[i]:

                        cv2.circle(img, (int(p1[i, 0, 0]), int(p1[i, 0, 1])), 2, (0, 0, 255), 2)
                        cv2.line(img, (int(subpixel_features[i, 0, 0]), int(subpixel_features[i, 0, 1])), (int(subpixel_features[i, 0, 0] + (p1[i][0, 0] - subpixel_features[i, 0, 0]) * 5),
                                                                   int(subpixel_features[i, 0, 1] + (p1[i][0, 1] - subpixel_features[i, 0, 1]) * 5)),
                                 (0, 0, 255), 2)

                subpixel_features = p1[st == 1].reshape(-1, 1, 2)
                index = index[st.flatten() == 1]

            # refresh features, if too many lost
            if len(subpixel_features) < 100:
                features = cv2.goodFeaturesToTrack(gray_scale, 200 - len(subpixel_features), 0.3, 7)
                new_subpixel_features = cv2.cornerSubPix(gray_scale, features, winSize, zeroZone, criteria=criteria)
                for i in range(len(new_subpixel_features)):
                    if np.min(np.linalg.norm((subpixel_features - new_subpixel_features[i]).reshape(len(subpixel_features), 2), axis=1)) > 10:
                        subpixel_features = np.append(subpixel_features, new_subpixel_features[i].reshape(-1, 1, 2), axis=0)
                        index = np.append(index, np.max(index) + 1)

            # update tracks
            for i in range(len(subpixel_features)):
                if index[i] in tracks:
                    tracks[index[i]][frame] = subpixel_features[i]
                else:
                    tracks[index[i]] = {frame: subpixel_features[i]}

            # visualise last frames of active tracks
            for i in range(len(index)):
                for f in range(frame - 20, frame):
                    if (f in tracks[index[i]]) and (f + 1 in tracks[index[i]]):
                        cv2.line(img,
                                 (int(tracks[index[i]][f][0, 0]), int(tracks[index[i]][f][0, 1])),
                                 (int(tracks[index[i]][f + 1][0, 0]), int(tracks[index[i]][f + 1][0, 1])),
                                 (0, 255, 0), 1)

            k = cv2.waitKey(10)
            if k % 256 == 27:
                print("Escape hit, closing...")
                break

            cv2.imshow("track feature", img)
            cv2.waitKey(0)

        cv2.destroyAllWindows()
        print("\n length of tracks: ", len(tracks))
        print("total number of frames: ", frame)

        return tracks, frame


# --------------------------------TASK 1 ------------------------------
# TASK 1 (pre-processing, 11 points)
def task1():
    matrix, image_points, object_points = task_1_AB()
    tracks, frame = task_1_BC()
    return tracks, matrix, frame, image_points, object_points

# --------------------------------TASK 1 End ------------------------------


# --------------------------------TASK 2 start ------------------------------

def calculate_epipoles(F):
    U,S,V = np.linalg.svd(F)
    e1 = V[2,:]

    U,S,V = np.linalg.svd(F.T)
    e2 = V[2,:]

    return e1,e2


def calculate_epipolar_line(F, x, width, height):
    l = np.matmul(F, x)
    l1 = np.cross([0,0,1],[width-1,0,1])
    l2 = np.cross([0,0,1],[0,height-1,1])
    l3 = np.cross([width-1,0,1],[width-1,height-1,1])
    l4 = np.cross([0,height-1,1],[width-1,height-1,1])
    x1 = np.cross(l,l1)
    x2 = np.cross(l,l2)
    x3 = np.cross(l,l3)
    x4 = np.cross(l,l4)
    x1 /= x1[2]
    x2 /= x2[2]
    x3 /= x3[2]
    x4 /= x4[2]
    result = []
    if (x1[0]>=0) and (x1[0]<=width):
        result.append(x1)
    if (x2[1]>=0) and (x2[1]<=height):
        result.append(x2)
    if (x3[1]>=0) and (x3[1]<=height):
        result.append(x3)
    if (x4[0]>=0) and (x4[0]<=width):
        result.append(x4)
    return result[0],result[1]


# reference LAB 9---
def extract_frames(filename, frames):
    result = {}
    camera = cv2.VideoCapture(filename)
    last_frame = max(frames)
    frame=0
    while camera.isOpened():
        ret, img= camera.read()
        if not ret:
            break
        if frame in frames:
            result[frame] = img
        frame += 1
        if frame>last_frame:
            break

    return result



# TASK 2 Main function
# TASK 2 (Fundamental matrix, 21 points)
def task2(tracks, frames):
    '''
    A)  Extract and visualise the feature tracks calculated in task 1 which are visible in both
        the first and the last frame to establish correspondences 𝒙𝒊 ↔ 𝒙𝒊′ between the two
        images [2 points]. Use Euclidean normalised homogeneous vectors.
    '''
    filename = "Assignment_MV_02_video.mp4"
    print("task2")

    firstFrame = 0
    lastFrame = frames

    # A) Stores the tracks visible in both first and last frame (common tracks)
    correspondences = []

    # Filter the tracks visible in both first and last frames
    for track in tracks:
        if firstFrame in tracks[track] and lastFrame in tracks[track]:
            x1 = [tracks[track][firstFrame][0, 1], tracks[track][firstFrame][0, 0], 1]
            x2 = [tracks[track][lastFrame][0, 1], tracks[track][lastFrame][0, 0], 1]
            correspondences.append([np.array(x1), np.array(x2)])

    images = extract_frames(filename, [0, lastFrame])

    # TASK 2-B
    '''
    1)  Calculate the mean feature coordinates in the first and last frame.
    2)  Calculate the corresponding standard deviations
    3)  Normalise all feature coordinates and work with 𝒚𝒊 = 𝑻𝒙𝒊 and 𝒚𝒊′ = 𝑻′𝒙𝒊′
        which are translated and scaled using the homographies
    '''

    meanfeatures = np.mean(np.array(correspondences)[:, 0, :2], axis=0)
    meanprime = np.mean(np.array(correspondences)[:, 1, :2], axis=0)
    print("Mean Feauters \n", meanfeatures)
    print("Mean Prime \n", meanprime)


    stddev = np.std(np.array(correspondences)[:, 0, :2], axis=0)
    stddevprime = np.std(np.array(correspondences)[:, 1, :2], axis=0)

    print("standard Deviation \n", stddev)
    print("stddevprime \n", stddevprime)

    Tran = np.array([[1/stddev[1], 0, -meanfeatures[1]/stddev[1]],
                  [0, 1/stddev[0], -meanfeatures[0]/stddev[0]],
                  [0, 0, 1]
                  ]
                 )

    TranPrime = np.array([[1 / stddevprime[1], 0, -meanprime[1] / stddevprime[1]],
                       [0, 1 / stddevprime[0], -meanprime[0] / stddevprime[0]],
                       [0, 0, 1]
                       ]
                      )


    # Normalise all feature coordinates --- 𝒚𝒊 = 𝑻𝒙𝒊 and 𝒚𝒊′ = 𝑻′𝒙𝒊′
    yi = (Tran@np.array(correspondences)[:, 0, :].T).T
    yiprime = (TranPrime@np.array(correspondences)[:, 1, :].T).T

    index = []
    for i in range(len(correspondences)):
        index.append(i)
    print("length of Indexs", len(index))

    a = [1, 0, 0]
    b = [0, 1, 0]
    c = [0, 0, 0]
    CorXX = np.array([a, b, c])

    bits = np.inf
    boc = len(correspondences)
    bestFundamentalmatrix = None
    bi = None
    k = 8

    # Iterating for 10000 times to find the best fundamental matrix
    i = 0
    while i in range(10000):
        outliers = list()
        inliners = list()
        inliners_test = 0


        random_eight_indices = random.sample(index, 8)
        left_indices = list(set(index) - set(random_eight_indices))

        A = np.zeros((0, 9))
        for y, yprime in zip(yi[random_eight_indices, :], yiprime[random_eight_indices, :]):
            ai = np.kron(y.T, yprime.T)
            A = np.append(A, [ai], axis=0)
            # print("-----ai------")
            # print(ai)

        U, S, V = np.linalg.svd(A)
        F = V[8, :].reshape(3, 3).T

        U, S, V = np.linalg.svd(F)
        F = np.matmul(U, np.matmul(np.diag([S[0], S[1], 0]), V))

        bestFundamentalmatrix = F

        # Task F
        inc = 0
        for x_i, x_i_dash in np.array(correspondences)[left_indices, :, :]:

            g_i = np.matmul(np.matmul(x_i_dash.reshape(3, 1).T, F), x_i.reshape(3, 1))
            variance = np.matmul(np.matmul(np.matmul(np.matmul(x_i_dash.reshape(3, 1).T, F), CorXX), F.T),
                                 x_i_dash.reshape(3, 1)) + np.matmul(
                np.matmul(np.matmul(np.matmul(x_i.reshape(3, 1).T, F.T), CorXX), F), x_i.reshape(3, 1))

            T_i = g_i / variance

            # print("g_i", g_i)
            # print("variance", variance)
            # print("T_i", T_i)

            if T_i > 6.635:
                outliers.append(inc)
            else:
                inliners.append([x_i, x_i_dash])
                inliners_test += T_i
            inc += 1

        if len(outliers) > 0:
            print("Outliers ", len(outliers), i + 1)

        if len(outliers) == boc:
            if inliners_test < bits:
                bits = inliners_test
                # print("discard previous fun matrix")
                bestFundamentalmatrix = F
                bi = inliners
        elif len(outliers) < boc:
            boc = len(outliers)
            bestFundamentalmatrix = F
            bi = inliners
        i+=1

    print("Outliers : ", boc)
    print(" Fundamental MAtrix", bestFundamentalmatrix)
    print("Inliners", len(bi))


    # TASK 2-H
    '''
    EPIPOLES
    '''
    e1, e2 = calculate_epipoles(bestFundamentalmatrix)
    print("------------- Epipoles---------------")
    print("----e1-------")
    print(e1)
    print("----e2-------")
    print(e2)
    print("----e1/e2-------")
    print(e1 / e1[2])
    print("----e2/e2-------")
    print(e2 / e2[2])

    '''---------------------------'''
    wid = images[firstFrame].shape[1]
    hei = images[lastFrame].shape[0]
    print("width", wid)
    print("heigth", hei)

    '''---------------------------'''
    x = np.array([0.5*wid, 0.5*hei, 1])

    # print("------------- Epipoles linear ---------------")
    x1, x2 = calculate_epipolar_line(bestFundamentalmatrix, x, wid, hei)

    cv2.circle(images[firstFrame], (int(x[0]/x[2]),int(x[1]/x[2])), 3, (0,255,0), 2)
    cv2.line(images[lastFrame], (int(x1[0]/x1[2]),int(x1[1]/x1[2])), (int(x2[0]/x2[2]),int(x2[1]/x2[2])), (0,255,0), 2)

    cv2.imshow("image 1", images[0])
    cv2.imshow("image 2", images[30])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return bi, bestFundamentalmatrix


# --------------------------------TASK 2 End ------------------------------


# --------------------------------TASK 3 start ------------------------------

def task3(bi, matrix, bestFundamentalmatrix, frame):
    print("# --------------------------------TASK 3 start ------------------------------")

    inliners = bi
    F = bestFundamentalmatrix
    K = matrix
    K_T = K.T
    numberofFrames = frame

    # task 3 A
    # print("-FM-",np.linalg.det(F))

    E_dash = np.matmul(np.matmul(K_T, F), K)
    print("Estimation of Essential_Matrix \n", E_dash)

    '''reference Lecture 11 slide 23'''
    U,S,V = np.linalg.svd(E_dash)

    # to verify the determinants of RM are not negative
    if np.linalg.det(V) < 0:
        V[2, :] *= -1

    if np.linalg.det(U) < 0:
        U[:, 2] *= -1

    # '''print all the values'''
    print("-----U------ \n", U)
    print("-----S------ \n", S)
    print("-----V------ \n", V)
    # print("Determinant_of_U \n", np.linalg.det(U))
    # print("Determinant_of_V \n", np.linalg.det(V))


    ''' Essential Matrix
        𝑬 = 𝑼 (𝜆, 𝜆, 0) 𝑽.T 
        where (𝜆, 𝜆, 0) it is a diagonal matrix
    '''
    first = S[0]
    second = S[1]
    lambda_= (first + second)/2

    E = np.matmul(U, np.matmul(np.diag([lambda_, lambda_, 0] ),V.T))
    print("Calculated Essential MAtrix is \n", E)

    # [[-114742.94092351 - 167061.34419041 - 71993.67860486]
    #  [204951.912187 - 71986.69990947 - 73875.28906867]
    #  [4874.60127611 - 103008.40412142 - 57129.59885998]]

    # [[-274772.366385 - 124414.32744542   71708.07628285]
    #  [188723.89705393 - 235724.6474957   134695.20188029]
    # [-77916.77233328 - 131140.15595583  75235.92467071]]

    ''' 
        Also make sure that the rotation 
        matrices of the singular value decomposition have positive determinants
    '''
    print("Positive determinants of Rotational MAtrix \n", np.linalg.det(E))

    # TASK 3 -B

    Z = np.array([
        [0,1,0], [-1,0,0], [0,0,0]
    ])

    W = np.array([
        [0,-1,0], [1,0,0], [0,0,1]
    ])

    speed_in_meters = (50*10)/36
    beta_ = (speed_in_meters * numberofFrames) / 30

    U_Z_U_T = (np.matmul(np.matmul(U, Z), U.T))

    S_RT_t1 = beta_ * U_Z_U_T
    S_RT_t2 = -S_RT_t1

    R_T_t1 = np.array([S_RT_t1[2,1], S_RT_t1[0,2], S_RT_t1[1,0]])
    R_T_t2 = np.array([S_RT_t2[2, 1], S_RT_t2[0, 2], S_RT_t2[1, 0]])

    print("----S_RT_t1----", S_RT_t1)
    print("----S_RT_t2----", S_RT_t2)
    print("----R_T_t1----", R_T_t1)
    print("----R_T_t2----", R_T_t2)

    #Rotational Matrix task 3 b
    R_T_1 = np.matmul(np.matmul(U, W), V.T)
    R_T_2 = np.matmul(np.matmul(U, W.T), V.T)
    Rotational_matrix = [R_T_1, R_T_2]

    print("---- Rotatinoal_Matrices_1----", R_T_1)
    print("---- Rotatinoal_Matrices_2----", R_T_2)

    # verifting for matrix is positive or not
    print("---Deter of Matrix_1", np.linalg.det(R_T_1))
    print("---Deter of Matrix_2", np.linalg.det(R_T_2))

    # Translation matrix task 3 b
    T_matrix_1 = np.matmul(np.linalg.inv(R_T_1), R_T_1)
    T_matrix_2 = np.matmul(np.linalg.inv(R_T_2), R_T_2)
    Translation_matrix = [T_matrix_1, T_matrix_2]

    print("---Translation matrix 1", np.linalg.det(R_T_1))
    print("---Translation matrix 1", np.linalg.det(R_T_2))

    '''
    Task 3 C
    '''
    total_coordinates_3d = {}
    total_counts = {}

    for rot_ind, rot_matrix in enumerate(Rotational_matrix):
        for tran_ind, tra_matrix in enumerate(Translation_matrix):
            count_best_points = 0
            coordinates_3d_points = list()
            for i1, i2 in inliners:
                ''' calculated in the previous subtask the directions 𝒎 and 𝒎′ '''
                m = np.matmul(np.linalg.inv(K), i1)
                m_dash = np.matmul(np.linalg.inv(K), i2)
                # print("--m--", m)
                # print("---m_dash---", m_dash)

                '''
                solving each components of the given linear equation
                𝑿[𝜆] = 𝜆𝒎
                𝑿[𝜇] = 𝒕 + 𝜇𝑹𝒎′
                calculate the unknown distances 𝜆 and 𝜇 by solving the linear equation system
                ( 𝒎𝑻𝒎 −𝒎𝑻𝑹𝒎′ )    𝜆         𝒕𝑻m
                                        =
                ( 𝒎𝑻𝑹𝒎′ −𝒎′𝑻𝒎′ )  𝜇         𝒕𝑻𝑹𝒎′
                to obtain the 3d coordinates of the scene points
                '''
                mTm = np.matmul(m.T, m)
                mTRm_dash = np.matmul(m.T, np.matmul(rot_matrix, m_dash))
                m_dash_TM_dash = np.matmul(m_dash.T, m_dash)
                tTm = np.matmul(tra_matrix.T, m)
                tTRm_dash = np.matmul(tra_matrix.T, np.matmul(rot_matrix, m_dash))

                print("----mTm----", mTm)
                print("----mTRm_dash----", mTRm_dash)
                print("----m_dash_TM_dash----", m_dash_TM_dash)
                print("----tTm----", tTm)
                print("----tTRm_dash----", tTRm_dash)


                '''𝜆 > 0 and 𝜇 > 0 '''
                lambda_mue = np.linalg.solve([[mTm, -mTRm_dash], [mTRm_dash, -m_dash_TM_dash]], [tTm, tTRm_dash])
                # print("-----lambda_mue--------",lambda_mue)
                if (lambda_mue[0] > 0).any() and (lambda_mue[1] > 0).any():
                    count_best_points += 1
                    x_lambda = lambda_mue[0] * m
                    x_mue = tra_matrix + np.multiply(lambda_mue[1], np.matmul(rot_matrix, m_dash))
                    coordinates_3d_points.append([x_lambda, x_mue])
                    # print("-----x_lambda--------", x_lambda)
                    # print("------x_mue-------", x_mue)
                    # print("-------coordinates_3d_points------", coordinates_3d_points)

            total_coordinates_3d[(rot_ind, tran_ind)] = coordinates_3d_points
            total_counts[(rot_ind, tran_ind)] = count_best_points

    print("---- total_coordinates_3d -----", total_coordinates_3d)
    print("---- total_counts -----", total_counts)

    bcc = max(list(total_counts.values()))
    bcci = np.argmax(list(total_counts.values()))
    bcck = list(total_counts.keys())[bcci]
    points_both_frame = total_coordinates_3d[bcck]
    best_rotational_matrix = Rotational_matrix[bcck[0]]
    best_translation_matrix = Translation_matrix[bcck[1]]

    print("----Best of Coordinates count-------", bcc)
    print("-----bcci------", bcci)
    print("-----bcck------", bcck)
    print("-----points_both_frame------", points_both_frame)
    print("----best_rotational_matrix-------", best_rotational_matrix)
    print("----best_translation_matrix-------", best_translation_matrix)
    # print("----array_points_both_frame-------", np.array((points_both_frame, np.uint16), dtype=object).shape)

    '''Project the 3d points into the first and the last frame and display their 
    position in relation to the corresponding features to visualise the reprojection error '''
    # numarray = np.array((points_both_frame, np.uint16), dtype=object)
    # ax = Axes3D(plt.figure())
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    #
    # x, y, z = np.array((points_both_frame, np.uint16), dtype=object)[:, 0, 0], np.array((points_both_frame, np.uint16), dtype=object)[:, 0, 1], np.array((points_both_frame, np.uint16), dtype=object)[:, 0, 2]
    # ax.scatter3D(x, y, z, marker='o', c='blue')
    #
    # ax.plot([0.], [0.], [0.], marker='X', c='blue')
    #
    # ax.plot(Translation_matrix[bcck[1]][0], Translation_matrix[bcck[1]][1], Translation_matrix[bcck[1]][2], marker='0', c='green')
    #
    # # np.array([study_minutes, np.zeros(100, np.uint16)], dtype=object)
    #
    # plt.show()








if __name__ == '__main__':


    '''
    reference for this assignment
    https://github.com/ShankarPendse/Machinevision/blob/main/scene_reconstruction.py
    '''
    # Task 1
    tracks, matrix, frame, image_points, object_points = task1()

    # Task 2
    bi, bestFundamentalmatrix = task2(tracks, frame)

    # # Task 3
    task3(bi, matrix, bestFundamentalmatrix, frame)