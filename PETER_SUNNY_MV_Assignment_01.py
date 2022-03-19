'''
    Full Name : Peter Sunny Shanthveer Markappa
    Student Id : R00208303
    Subject : Machine Vision
    Assignment Number : 01
    Submission Date : 18-March-2022
'''


# importing the libraries
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def display_result(value, image):
    cv2.imshow(value, image)
    # filename = value+'.png'
    # cv2.imwrite(filename, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_with_sigma(value, sigma, image):
    cv2.imshow(value %sigma, image)
    # filename = value+'.png'
    # cv2.imwrite(filename %sigma, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_result(value, result):
    print(value, result)
    print("------------------------")


# ---------------------------------------------------------------------------------------------------
# ------------------------------------TASK 1(a) -----------------------------------------------------
# ---------------------------------------------------------------------------------------------------
def load_image():
    '''  Task 1(a) -- task-- load, convert and return grey image with float 32
        # loading the image in single channel grey scale format
        # 1 represent color and 0 represent grey
        # Here the image is loaded and converted into grey scale by passing value 0
    '''
    input_image = cv2.imread("Assignment_MV_1_image.png")

    # converting the datatype from uint8 ie., 8bits to float 32
    input_image = input_image.astype(np.float32)
    greyImage = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

    # # ------Printing ------------
    print_result("Input Image Data Type = ", input_image.dtype)
    print_result("Grey Image Data Type = ", greyImage.dtype)
    print_result("Grey Image Shape = ", greyImage.shape)

    # # ------ Displaying ------------
    display_result('Task 1-A-Input-Image-uint8', input_image/255)
    display_result('Task 1-A-Grey-Image-float32', greyImage/255)

    return greyImage, input_image


# ---------------------------------------------------------------------------------------------------
# ------------------------------------TASK 1(B) -----------------------------------------------------
# ---------------------------------------------------------------------------------------------------

def gaussianSmoothingKernels(greyImage):
    '''  Task 1(b) -- TASKS
    Create twelve Gaussian smoothing kernels with increasing scales plot each of these kernels as image
    Apply these kernels to the input image to create a scale-space representation

    '''
    gaussianKernals, filterImages = [], []

    # creating the list to hold all the sigma value
    sigmaList = [(2 ** (k / 2)) for k in range(12)]

    'Each loop create the kernal using the sigma value from the list generated' \
    'than it will display the kernal and the same kernal will be applied to input image' \
    'each loop will display the kernal and once you close the figure it will generate the scale space image with respect to the kernal'

    for sigma in sigmaList:
        # totally 12 smoothed images will be created
        # reference from lab solution
        x, y = np.meshgrid(np.arange(-3 * sigma, 3 * sigma),
                           np.arange(-3 * sigma, 3 * sigma))

        gaussianXY = (1 / (2 * np.pi * sigma ** 2)) * (np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)))
        gaussianKernals.append(gaussianXY)
        print_result("gaussianXY.shape = ", gaussianXY.shape)

        # # ---------- Displaying the kernal --------------
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, gaussianXY)
        plt.title("Gaussian Kernal for %s" % sigma)
        plt.figure()
        plt.imshow(gaussianXY)
        plt.title("Gaussian Kernal for %s" % sigma)
        plt.show()

        # -----------Applying the input image to the gaussian kernals-----------
        filterImg = cv2.filter2D(greyImage, -1, gaussianXY)

        '''---------------Display all the Filter Images---------------------'''
        display_with_sigma("Scale Spaced Images with %s", sigma, filterImg/255)

        filterImages.append(filterImg)

    # ------------Returning the list of gaussian kernals, 12 gaussian smoothed images and sigma list-----------
    return gaussianKernals, filterImages, sigmaList



# ---------------------------------------------------------------------------------------------------
# ------------------------------------TASK 2(ABC) -----------------------------------------------------
# ---------------------------------------------------------------------------------------------------

def DOG_KeyPoints(gaussianKernals, filterImages, sigmaList, greyImage, input_image):
    gaussian_difference_images, gaussian_difference_sigma = [], []

    print_result("Length of Gaussian Kernals = ", len(gaussianKernals))

    for i in range(0, len(gaussianKernals)-1):
        # here we will find difference of gaussian image
        DOG = cv2.subtract(filterImages[i], filterImages[i+1])

        gaussian_difference_sigma.append(min(sigmaList[i], sigmaList[i+1]))
        gaussian_difference_images.append(DOG)

        '''----------------Display the DOG Images-----------------'''
        display_result("Difference of Gaussian : ", DOG)

    #------------------------------------- TASK 2 B ------------------------------
    T, keyPoints, i = 10, [], 0

    while i in range(0, len(gaussian_difference_images)):
        currentImage = gaussian_difference_images[i]

        # Here we are checking the condition for scale images for comparision.
        ''' ---- if the current image is the first image, then the previous scale space is considered as current image
            (i.e., first image) but by making all the values to zeros ----- '''
        if i == 0:
            preImage = np.zeros(currentImage.shape)
        else:
            preImage = gaussian_difference_images[i-1]

        ''' ---- if the current image is the last image, then the Next scale space is considered as current image
            (i.e., last image) but by making all the values to zeros ----- '''
        if i == len(gaussian_difference_images)-1:
            nextImage = np.zeros(currentImage.shape)
        else:
            nextImage = gaussian_difference_images[i+1]

        '''
            total there will be 26 pixel in total ---
            for present image 8 and two other scale space 9 and 9 
            considering present value x,y then it will be compared with 8 of its own and 9 of previous and 9 of next adjacent scale
        '''
        # reference from the lab solution
        for x in range(1, currentImage.shape[0]-1):
            for y in range(1, currentImage.shape[1]-1):
                if ((currentImage[x, y] > T) and
                        (currentImage[x, y] > currentImage[x - 1, y - 1]) and
                        (currentImage[x, y] > currentImage[x - 1, y]) and
                        (currentImage[x, y] > currentImage[x - 1, y + 1]) and
                        (currentImage[x, y] > currentImage[x, y - 1]) and
                        (currentImage[x, y] > currentImage[x, y + 1]) and
                        (currentImage[x, y] > currentImage[x + 1, y - 1]) and
                        (currentImage[x, y] > currentImage[x + 1, y]) and
                        (currentImage[x, y] > currentImage[x + 1, y + 1]) and
                        (currentImage[x, y] > preImage[x, y]) and
                        (currentImage[x, y] > preImage[x, y + 1]) and
                        (currentImage[x, y] > preImage[x, y - 1]) and
                        (currentImage[x, y] > preImage[x - 1, y]) and
                        (currentImage[x, y] > preImage[x + 1, y]) and
                        (currentImage[x, y] > preImage[x + 1, y + 1]) and
                        (currentImage[x, y] > preImage[x - 1, y + 1]) and
                        (currentImage[x, y] > preImage[x + 1, y - 1]) and
                        (currentImage[x, y] > preImage[x - 1, y - 1]) and
                        (currentImage[x, y] > nextImage[x - 1, y + 1]) and
                        (currentImage[x, y] > nextImage[x, y + 1]) and
                        (currentImage[x, y] > nextImage[x + 1, y + 1]) and
                        (currentImage[x, y] > nextImage[x - 1, y]) and
                        (currentImage[x, y] > nextImage[x, y]) and
                        (currentImage[x, y] > nextImage[x + 1, y]) and
                        (currentImage[x, y] > nextImage[x - 1, y - 1]) and
                        (currentImage[x, y] > nextImage[x, y - 1]) and
                        (currentImage[x, y] > nextImage[x + 1, y - 1])):
                        keyPoints.append({
                            'xcor': x,
                            'ycor': y,
                            'sigma': gaussian_difference_sigma[i]
                        })
        i+=1


    # ------------------------------------------Task 2 C------------------------------------------------
    for k in keyPoints:
        x, y, sigma = k['xcor'], k['ycor'], k['sigma']
        cv2.circle(input_image, (y, x), int(3*sigma), (0,0,255), thickness=1)
    display_result("Key Points Result :", input_image/255)

    return keyPoints



# ---------------------------------------------------------------------------------------------------
# ------------------------------------TASK 3(ABC) ---------------------------------------------------
# ---------------------------------------------------------------------------------------------------

def scaleSpaceImageDerivative(filterImages, sigmaList, keypoints, input_image):
    '''
        :param filterImages:
        :param sigmaList:
        :param keypoints:
        :return:

    TODO:   1) Calculate derivatives of all scale-space images
                dx = (1 0 −1)  dy = (1 0 −1)T
            2) calculate gradient lengths and directions
            3) Calculate gaussian weighting function .. calculate for each 10 degree range −180 ≤ 𝑖 < 180
            4) visualise the orientation of all key points
    '''

    dx = np.array([[1, 0, -1]])
    dy = np.array([[1, 0, -1]]).T
    der_img, keypointsOrientation = [], []

    for ind, img in enumerate(filterImages):
        # reference from the lab solution
        # dog_x = cv2.filter2D(img.astype('float'), -1, dog_kernel_x)
        # dog_y = cv2.filter2D(img.astype('float'), -1, dog_kernel_y)
        derivative_x_ = cv2.filter2D(img, -1, dx)
        derivative_y_ = cv2.filter2D(img, -1, dy)

        # Storing the derivative_x and derivative_y and their sigma value in dictionary
        der_img.append({
            'sigma': sigmaList[ind],
            'derivative_x_': derivative_x_,
            'derivative_y_': derivative_y_
        })

        # Display the resulting derivative images 𝑔𝑥 and 𝑔𝑦 at all scales
        display_with_sigma("Derivative of dx for %sigma", sigmaList[ind], derivative_x_)
        display_with_sigma("Derivative of dy for %sigma", sigmaList[ind], derivative_y_)


    # ---------------------- TASK 3-B -------------------------------

    # total 36 bins will
    bins = np.arange(-180, 180, 10)  # bin values will be between -180 to +180 degree = 360 degree with 10 degree range

    for k in keypoints:
        try:
            # (q, r) ∈ { 3/2 kσ | k = −3, . . ,3} × { 3/2 kσ | k = −3, . . ,3}
            # x, y = np.meshgrid(np.arange(6 * sigma_d), np.arange(6 * sigma_d))
            q, r = np.meshgrid(
                (3/2) * np.linspace(-2, 2, 7) * k['sigma'],
                (3/2) * np.linspace(-2, 2, 7) * k['sigma'],
            )
        # now we have to get relative coordinate of the grid with respect to key point x,y coordinates
        # Use nearest neighbour interpolation to sample the gradient grid
        # # now we have to get derivatives of gaussian

            gaussian_derivative = []
            # we are getting the dx and dy (derivative images) with same scale of current key pont
            for s in der_img:
                if k['sigma'] == s['sigma']:
                    gaussian_derivative = s

            # mqr = √𝑔𝑥2[𝑥 + 𝑞, 𝑦 + 𝑟] + 𝑔𝑦2[𝑥 + 𝑞, 𝑦 + 𝑟]
            x = gaussian_derivative['derivative_x_'][np.round(q + k['xcor']).astype(int), np.round(r + k['ycor']).astype(int)]
            y = gaussian_derivative['derivative_y_'][np.round(q + k['xcor']).astype(int), np.round(r + k['ycor']).astype(int)]

            # Gradient Length
            gradient_length = np.sqrt(np.square(x) + np.square(y))

            #  gradient direction
            # θqr = 𝑎𝑡𝑎𝑛2 [𝑔𝑦 [𝑥 + 𝑞, 𝑦 + 𝑟], 𝑔𝑥 [𝑥 + 𝑞, 𝑦 + 𝑟]]
            thetadegree = np.rad2deg(np.arctan2(y, x))


            #---------- TASK 3-C ------------------

            # Calculate a Gaussian weighting function
            gaussian_weiging_function = (np.exp(-(q**2 + r**2) / ((9 * k['sigma']**2) / 2))) / ((9 * np.pi * k['sigma']**2) / 2)

            # weighted gradient lengths wqr + mqr
            weightedgradientlength = gaussian_weiging_function * gradient_length

            # calculating the index of the histogram using digitize then adding it
            histogram_sum = np.bincount(np.digitize(thetadegree.flatten(), np.arange(-180, 180, 10)).flatten(), weightedgradientlength.flatten(), minlength=36)

            # Use the maximum of this orientation histogram ℎ to determine the dominant orientation
            orentation_histogram_angle = (2 * np.pi / 36) * (0.5 + float(bins[np.argmax(histogram_sum)]))

            # storing all the x-coorginte, y-coordinate with its sigma value and the angle theta in the dictionary for task 4
            keypointsOrientation.append(
                {
                    'xcor': k['xcor'],
                    'ycor': k['ycor'],
                    'sigma': k['sigma'],
                    'thetaangle': orentation_histogram_angle
                }
            )

        except Exception:
            continue


    # #--------------- TASK 3 D ----------------------
    for k in keypointsOrientation:
        x, y, sigma, theta = k['xcor'], k['ycor'], k['sigma'], k['thetaangle']

        cv2.circle(input_image, (y, x), int(3*sigma), (0,0,255), thickness=1)
        cos = int(np.round(np.cos(theta) * int(3 * sigma)))
        sin = int(np.round(np.sin(theta) * int(3 * sigma)))
        cv2.line(input_image, (y, x), (y + cos, x + sin), (0, 255, 255), 2)

    '''----------- Displaying the Key Point Orientation ---------------------------'''
    display_result("key point orientations", input_image/255)

    return der_img, keypointsOrientation


# ---------------------------------------------------------------------------------------------------
# ------------------------------------TASK 4(ABC) ---------------------------------------------------
# ---------------------------------------------------------------------------------------------------

def featuredescriptors(der_img, keypointsOrientation, input_image):
    '''
    :param der_img: takes the dictionary of derivative images with x, y, and sigma value
    :param keypointsOrientation: takes dictionary of key point orientation
    :return: returns nothing where as printing all the statement here only
    '''
    feature_key_descriptor = []
    bins = np.arange(0, 360, 45)
    featuredescriptors_, normalisedfeaturedescriptors_, capped_ = [], [], []

    i = 1
    for k in keypointsOrientation:
        try:
            keypointhist = []
            gaussian_derivative = {}

            # we will get dx and dy of gaussian derivate images as same scale as present key point
            for s in der_img:
                if k['sigma'] == s['sigma']:
                    gaussian_derivative = s

            sigma = k['sigma']
            # {−2, …, 1} × {−2, …, 1} covering an area of ±9/2𝜎 are given by
            i = -2
            while i<2:
                # for i in range(-2, 2, 1):
                j=0
                while j<4:
                    # for j in range(4):
                    s, t = np.meshgrid(
                        (3 / 16) * np.linspace(-3, 3, 4) * (4 * i + j + (1 / 2)) * sigma,
                        (3 / 16) * np.linspace(-3, 3, 4) * (4 * i + j + (1 / 2)) * sigma
                    )
                    x, y = k['xcor'], k['ycor']
                    sx = np.round(s + x).astype(int)
                    ty = np.round(s + y).astype(int)

                    # Gaussian weighting function
                    GaussianWeighted = (np.exp(-((s ** 2) + (t ** 2)) / ((81 * np.pi * (sigma ** 2)) / 2))) / (
                            (81 * np.pi * (sigma ** 2)) / 2)

                    # gradient lengths
                    x = gaussian_derivative['derivative_x_'][sx, ty]
                    y = gaussian_derivative['derivative_y_'][sx, ty]
                    gradientLength = np.sqrt(np.square(x) + np.square(y))

                    # weighted gredient length subgrid 4 by 4 calculation
                    totalgradientlength = GaussianWeighted * gradientLength

                    # gradient directions
                    # to deal with negative values we have use %360
                    th = (np.rad2deg(np.arctan2(gaussian_derivative['derivative_y_'][sx, ty],
                                                gaussian_derivative['derivative_x_'][sx, ty])) % 360)
                    kth = (k['thetaangle'] % 360)
                    thetadegree = th - kth

                    # digitize will return the indices of all orientation
                    # 0, 360, 45 is histogram bin where the value will be betwen 0 to 360 with difference of 45 degree
                    indices = np.digitize(np.abs(thetadegree).flatten(), np.arange(0, 360, 45)) - 1

                    histogram_sum = np.bincount(np.digitize(thetadegree.flatten(), np.arange(0, 360, 40)).flatten(),
                                                totalgradientlength.flatten(), minlength=8)
                    orentation_histogram_angle = (2 * np.pi / 36) * (0.5 + float(bins[np.argmax(histogram_sum)]))

                    # histogram is appended here to create 16x16
                    keypointhist.append(np.bincount(indices.flatten(), totalgradientlength.flatten(), minlength=8))

                    feature_key_descriptor.append({
                        'xcor': k['xcor'],
                        'ycor': k['ycor'],
                        'sigma': k['sigma'],
                        'thetaangle': orentation_histogram_angle
                    })


                    j+=1
                i+=1

        except Exception:
            continue

        # ------------------------ task 4 C -----------------------
        # Concatenate all these 16 histogram 8 - vectors into a single 128 - vector
        featuredescriptors = np.concatenate(keypointhist)

        # Normalise this descriptor vector dividing it by its length
        normalisedfeaturedescriptors = featuredescriptors / np.sqrt(np.dot(featuredescriptors.T, featuredescriptors))

        # by capping the  maximum value of the vector at 0.2
        capfeature = np.clip(normalisedfeaturedescriptors, a_min=np.min(normalisedfeaturedescriptors), a_max=0.2 )

        featuredescriptors_.append(featuredescriptors)
        normalisedfeaturedescriptors_.append(normalisedfeaturedescriptors)
        capped_.append(capfeature)


        '''-----------------------------Print the results---------------------------'''
        '''
            the below displayed result gives :
            ------------------------ 
            Feature Descriptors =  128
            ------------------------
            Normalized Feature Descriptors =  128
            ------------------------
            Feature descriptor after capping = 128
            ------------------------
            because 16 histogram with 8-vectors are concatenated to single 128-vector
        '''
        print_result("Feature Descriptors = ", len(featuredescriptors))
        print_result("Normalized Feature Descriptors = ", len(normalisedfeaturedescriptors))
        print_result("Feature descriptor after clippting", len(capfeature))

    print_result("Total Length of key point Feature Descriptor = ", len(featuredescriptors_))
    print_result("Total Length of Normalised Feature Descriptor = ", len(normalisedfeaturedescriptors_))
    print_result("Total Length of Capped Feature Descriptor = ", len(capped_))

    # Displaying feature descriptors
    for k in feature_key_descriptor:
        x, y, sigma, theta = k['xcor'], k['ycor'], k['sigma'], k['thetaangle']

        cv2.circle(input_image, (y, x), int(3 * sigma), (0, 0, 255), thickness=1)
        cos = int(np.round(np.cos(theta) * int(3 * sigma)))
        sin = int(np.round(np.sin(theta) * int(3 * sigma)))
        cv2.line(input_image, (y, x), (y + cos, x + sin), (255, 0, 255), 2)

    '''----------- Displaying the Key Point Orientation ---------------------------'''
    display_result("Feature Descriptor ", input_image/255)


    print("*********************  END ****************************")





def main():
    # --------------------------------------------------------------------------
    # -------------------------TASK 1 ------------------------------------------
    # --------------------------------------------------------------------------
    greyImage, input_image = load_image()  # task 1(A)
    gaussianKernals, filterImages, sigmaList = gaussianSmoothingKernels(greyImage)  # Task 1(B)


    # --------------------------------------------------------------------------
    # -------------------------TASK 2 ------------------------------------------
    # --------------------------------------------------------------------------
    keypoints = DOG_KeyPoints(gaussianKernals, filterImages, sigmaList, greyImage, input_image)


    # --------------------------------------------------------------------------
    # -------------------------TASK 3 ------------------------------------------
    # --------------------------------------------------------------------------
    der_img, keypointsOrientation = scaleSpaceImageDerivative(filterImages, sigmaList, keypoints, input_image)


    # --------------------------------------------------------------------------
    # -------------------------TASK 4 ------------------------------------------
    # --------------------------------------------------------------------------
    featuredescriptors(der_img, keypointsOrientation, input_image)


    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------
# ------------------------- main -------------------------------------------
# --------------------------------------------------------------------------

if __name__ == "__main__":
    main()