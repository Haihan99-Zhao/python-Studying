# Define all functions in this module.
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from datetime import datetime
from pathlib import Path
import os
import random


# *------------------------------------------------
# Task 1: Reading images into NumPy arrays [3 marks]
# *-----------------------------------------------

def read_image(path, show=False):

    """
    This function aim to read a picture file through specific store path
    into a 101*101*3/4 array for further process. And based on the number
    of color channels, determine whether the file is of RGB(3) or RGBA(4). 

    Moreover, if user needs to output plots, they can modify it in the "show" 
    option. Then, we will draw new plots with differen backgrounds.

    Finally, this function returns the Numpy array.
    """

    image = plt.imread(path) # read a file from "path" and
                             # automatically convert to an array

    # If we need "show" the pictures
    if show :
        # Create a canvas that can hold images from all channels.
        fig, ax = plt.subplots(1, image.shape[2]+1, figsize=(10,8)) 

        # The first position shows the original look of the file image, 
        ax[0].imshow(image)

        # with a title added and the axes turned off.
        ax[0].set_title("The original plot")
        ax[0].axis('off')

        # Next, we will generate images under different channal backgrouds.
        # The steps are to set the initial values of the other channal to 0
        # (white) and only retain the values of the specific channel.
        color_set = ["Red", "Green", "Blue", "Alpha"]

        # Generate a new background image based on channels.
        for i in range(image.shape[2]):
            tmpt_image = np.zeros_like(image) # with same dimensions filled with 0.
            tmpt_image[:, :, i] = image[:, :, i] # retain values

            #Same process like above
            ax[i+1].imshow(tmpt_image)
            ax[i+1].set_title(f"The {color_set[i]} plot")
            ax[i+1].axis('off')

        plt.show() # Show plots!!!

        # Determine the file type based on the third dimension of the matrix,
        # which refers to the number of elements in each vector.
        if image.shape[-1] == 3:
            print(f' The format of file is RGB.\n')
        else:
            print(f' The format of file is RGBA.\n')

    return image 


# **------------------------------------------------
# Task 2: Clean up the images to extract data [8 marks]
# **-----------------------------------------------

"""
# Under normal circumstances, the images we obtain will not be as clean as the test
# images, which may not contain noise.Therefore, before executing this function, a 
# noise-cleaning step will be performed. However, this step should only be executed if 
# the image contains noise.
"""


# Extra function: Using outliers to clean noise points.
def outliers_clean_points(clock_RGB, n=2):

    """
    This function designed for removing "noise", the method is that
    for color points, compared to gray points and noise, the value
    differences between channels are relatively large, which also leads
    to a higher variance.

    Therefore, we can identify color points as outliers based on the 
    self-standard-deviation feature and remove all the noise points.
    """

    # First, we calculate the variance within each pixel.
    each_std = np.std(clock_RGB, axis=2)

    # Flatten the matrix into a one-dimensional vector only for calculation. 
    reshaped_std = each_std.reshape(-1)

    # Calculate the standard deviation and mean, and set the outlier boundary,
    # defaulting to 6 times the standard deciation.
    total_std = np.std(reshaped_std)
    total_mean = np.mean(reshaped_std)
    upper_bound = total_mean + n * total_std

    # Intialize the background of the new image as white.
    new_image_array = np.ones_like(clock_RGB)

    # Retain the values from the old matrix meet the outlier criteria in new one.
    for i in range(clock_RGB.shape[0]):
        for j in range(clock_RGB.shape[1]):
            pixel = clock_RGB[i, j]

            pixel_variance = np.std(pixel) # Calculate self-standard-deviation
            if pixel_variance >= upper_bound:
                new_image_array[i,j] = pixel

    return new_image_array



 
def get_clock_hands(clock_RGB):

    """
    This function aims to figure out red points(hour hand) and green ones(minute hand),
    for "R", "G", "B" channels, we check if the difference with other channels exceeds
    a certain threshold to determine if a point is a color point

    e.g. To find the red points, the value of the first channel(R) will exceed a certain
         threshold difference compared to the other channels.

    Notice: we assume this function operate under clean picture!!!
    """
    
    # Intailize values
    hour_hand = []
    minute_hand = []
    threshold = 0.1
    
    clock_RGB = outliers_clean_points(clock_RGB)

    # Identify the corresponding hour hand by comparing the red channel(first) and muntute
    # hand by the green channel(second).
    for x in range(clock_RGB.shape[0]):
        for y in range(clock_RGB.shape[1]):
            vec = clock_RGB[x,y]
            index = [x,y]
            if vec[0] > vec[1] + threshold  and vec[0] > vec[2] + threshold:
                hour_hand.append(index)
            elif vec[1] > vec[0] + threshold and vec[1] > vec[2] + threshold:
                minute_hand.append(index)

    return hour_hand, minute_hand



# ***------------------------------------------------
# Task 3: Calculate the angle of the two hands [8 marks]
# ***------------------------------------------------


# Extra function: Find the new origin and move all coordinates.
def change_original_point(coords):
    """
    The purpose of this function is to move all coords to a new coordinate system (new origin).
    It consists of the following steps: 

    1. Obtain a rough center point with 10% deviation

    2.The coordinate closest to the center point becomes the new origin, and all coordinates 
    are moved
    """
    #-----------------------------------------------------------
    # *Step1: create center point with 10% deviation
    obscure_point = np.zeros((2))
    obscure_point[0] = (50* (1 - random.uniform(-0.1, 0.1)))
    obscure_point[1] = (50* (1 - random.uniform(-0.1, 0.1)))

    #-----------------------------------------------------------
    # **Step2:calculate the Mahalanobis distance from each coordinate to the center point.
    distance = np.linalg.norm(coords - obscure_point, axis=1)

    original_point = coords[distance.argmin()] # find the closest
    new_coords = coords - original_point # move all coordinates

    return new_coords


def get_angle(coords):

    """
    This function uses the input clock hand array to determine position and calculate
    angles (based on the vertical line as 12 o'clock).

    First, We check whether the set of coordinates is of array type. If it is not, 
    an error will be raised.

    Next, we use above extra function to move all the points. Then we use this given 
    new set of coordinates to determine the quadrant.

    Finally, we first discussed the cases where the tangent values are undefined for
    the 3 o'clock and 9 o'clock positions. Then, based on the quadrant, we calculated
    the correct angle (in radians) by considering the direction of the positive axis
    """

    if isinstance(coords, np.ndarray):
        True
    else:
        coords = np.array(coords)
        # raise Warning("The input is not of Numpy array type, please try again")
    
    pixels_x = coords[:, 0] # Extract the original x-coordinate.
    pixels_y = coords[:, 1] # Extract the original y-coordinate.

    changed_coords = change_original_point(coords) # Move the point to the new origin.

    changed_pixels_x = changed_coords[:, 0] # Extract the new x-coordinate.
    changed_pixels_y = changed_coords[:, 1] # Extract the new y-coordinate.

    # The sign of x-coordinate of the point farthest from the origin can help us determine 
    # in which quadrant the hand-endpoint lies. This will assist in discussing the 
    # situation based on different cases.

    # The "argmax()" function helps us find the index of that point,
    # which helps find the point with the farthest x-value.
    sign_max_absx = changed_pixels_x[np.abs(changed_pixels_x).argmax()]

    # *Special situations:
    # Horizontal clock hand: 3 o'clock and 9 o'clock.
    if len(np.unique(pixels_x)) == 1:
        
        # Using median to determine if the hand is at 9 o'clock position.
        if np.median(changed_pixels_y) < 0:
            radian = 3 * np.pi / 2

        # determine if 3 o'clock position.
        else:
            radian = np.pi / 2

    # Vertical clock hand: 12 o'clock and 6 o'clock.
    elif len(np.unique(pixels_y)) == 1:

        # determine if 12 o'clock position.
        if np.median(changed_pixels_x) < 0:
            radian = 0

        # determine if 6 o'clock position.
        else:
            radian = np.pi

    # Normal situations:
    # Firstly, obtain the slope of the coordinates using linear fitting.
    else:

        # The first return value of function linregress is slope.
        # And the result is the same whether fitting with the coordinates
        # before or after the changed origin.
        slope = linregress(pixels_x,pixels_y)[0]


        if np.arctan(slope) > 0 : # Determine the first and third quadrant

            if sign_max_absx > 0: # the first quadrant
                radian = np.pi - np.arctan(slope)

            elif sign_max_absx < 0: # the third  quadrant
                radian = 2*np.pi - np.arctan(slope)
        else: 
            if sign_max_absx > 0: # the second quadrant
                radian = np.pi + abs(np.arctan(slope))

            elif sign_max_absx < 0: # the forth quadrant
                radian = abs(np.arctan(slope))
    
    return radian # back radians



# ****------------------------------------------------
# Task 4: Analog to digital conversion [5 marks]
# ****------------------------------------------------

def analog_to_digital(angle_hour, angle_minute):

    """
    This function will operate the exact time when input two angles for hour and munute.

    First, we will determine the time based on the relationship between the angle and 
    the  clock's hand positions. 

    Then, we will adjust the standard format so that both the hour and minute are displayed
    as two digits. And change 0 o'clock to 12 o'clock
    """
    # Convert it to a "float" type for calculation
    angle_hour,angle_minute = float(angle_hour),float(angle_minute)

    # A circle(2*pi) is evenly divided by 12 hours, so each one hour occupies pi/6 radians.
    # A circle(2*pi) is evenly divided by 60 minutes, so each one minute occupies pi/30 radians.
    time_hour = int(angle_hour / (np.pi/6)) # "int" helps us reatian only the integer part 
    time_minute = int(angle_minute / (np.pi/30))
    
    # When the time is less than one hour, we we consider it to be shortly after 12 o'clock
    if time_hour == 0:
        time_hour = 12
    
    # Convert to "string" and pad it to be a two-digit number
    modified_hour = f"0{time_hour}"
    modified_minute = f"0{time_minute}"
    
    return f"{modified_hour[-2:]}:{modified_minute[-2:]}" # Only output the last two digits.




# *****------------------------------------------------
# Task 5: Find the misalignment [5 marks]
# *****------------------------------------------------

def check_alignment(angle_hour, angle_minute):
    """
    This function is designed to calculate the difference between the clock's displayed time
    and the real time. The steps are as follows:

    1. Use the hour hand to determine the real time

    2. Read the positions of hour and minute hands separately to get the displayed time

    3. These readings are then compared to compute the time difference.
    """
    
    # Convert it to a "float" type for calculation
    angle_hour,angle_minute = float(angle_hour),float(angle_minute)

    #-----------------------------------------------------------
    # *Step1: figure out real time only through hour hand
    
    correct_hour = np.floor(angle_hour / (np.pi / 6)) #The integer part represents the current hour.

    # The decimal part refers to the proportion of the hour(60mins) that corresponds to the
    # current minutes.
    correct_minute = ((angle_hour / (np.pi/6)) - correct_hour) * 60

    #-----------------------------------------------------------
    # **Step 2: reading minute hand to get the displayed time directly.
    wrong_minute = angle_minute / (np.pi/30)
    
    #-----------------------------------------------------------
    # ***Step 3: alignment can be calbulated by difference.
    diff = wrong_minute - correct_minute

    # Since the error will not exceed 30 minutes. 
    # e.g.If the clock is slow by 40 minutes, we interpret it as being fast by 20 minutes.
    if abs(diff) > 30: 
        if diff <0: # view fast as slow, so plus an hour
            return diff+60
        else:       # view slow as fast, so minus an hour
            return diff-60
    else:
        return diff



# *****------------------------------------------------
# Task 6: Reporting to the quality team [6 marks]
# *****------------------------------------------------

def validate_batch(folder_path, tolerance):
    """
    This function is used to recongnize clock images and provide basic information about these.
    It will utimately output a ".txt" text file with analyzing misaligment if exceeds tolerance,
    saved in a newly created folder. The process is divided into following steps:

    1. Prepare the relevant parameters to be output, such as batch number, generation time, etc.

    2. Edit the output text to include the necessary information.

    3. To write the text content and analyze the images sequentially.
    """

    #-----------------------------------------------------------
    # *Step1: prepare the parameters and read the paths and their contents

    number_batch = folder_path[-1] # the last character of the path as the batch number

    # store the current time and convert it to a specific format
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d, %H:%M:%S") 

    folder = Path(folder_path) # set the path for reading the clock images
    files = list(folder.glob("*.png")) # set the path for reading all ".png" files from a folder
    check_sum = len(files) # calculate the number of files in the batch
    
    # Initializing...
    abnormal_sum = 0
    abnormal_list = []
    abnormal_diffs = []
    
    for clock in files:
        clock_array = read_image(clock) # get a Numpy array from a "png" file
        # new_array = outliers_clean_points(clock_array) # clean all nosie
        hour_hand, minute_hand = get_clock_hands(clock_array) # get all the points of clock hands

        hour_angle = get_angle(hour_hand) # get the angle of hour hand
        minute_angle = get_angle(minute_hand) # get the angle of minute hand

        diff = int(check_alignment(hour_angle, minute_angle)) # calculate the error and retain intergal

        if abs(diff) > tolerance: # if exceeds the set range...

            abnormal_name = clock.stem # get the file name directly
            abnormal_sum += 1 # calculate total number of abnormal clock
            abnormal_list.append(abnormal_name) # record their names
            abnormal_diffs.append(diff) #record their misalignment
    
    # calculate the proportion of abnormal clocks and convert it into percentage
    rate_batch = round((1 - (abnormal_sum/check_sum)) * 100, 1) 
    if rate_batch == 100.0:
        rate_batch = 100
    elif rate_batch == 0.0:
        rate_batch = 0
    # Creat the keys and values of a new dict: their names and errors
    abnormal_dict = dict(zip(abnormal_list, abnormal_diffs))
    # sort the contents of a dictionary based on the abs of the error
    sorted_abnormal = dict(sorted(abnormal_dict.items(), key=lambda item: abs(item[1]), reverse=True))

    #-----------------------------------------------------------
    # **Step2: wirte down the main text content(title)
    report_main = [
        f"Batch number: {number_batch}\n"
        f"Checked on {formatted_datetime}\n"
        "\n"
        f"Total number of clocks: {check_sum}\n"
        f"Number of clocks passing quality control ({tolerance}-minute tolerance): {check_sum - abnormal_sum}\n"
        f"Batch quality: {rate_batch}%\n"
        "\n"
    ]

    #-----------------------------------------------------------
    # ***Step3: write a new text file in a new created folder
    
    if not os.path.exists("QC_reports"): # If the folder does not exist, create one.
        os.makedirs("QC_reports")

    # Add the text file name to the directory.
    report_file_path = os.path.join("QC_reports", f'batch_{number_batch}_QC.txt')

    with open(report_file_path, 'w') as file: # wirte content...
        file.writelines(report_main) # write the main
        if rate_batch != 100:
            file.write(f"Clocks to send back for readjustment:\n")
        for clock_label, clock_diff in sorted_abnormal.items(): # write the abnormal datas
            if clock_diff > 0: 
                clock_diff = f"+{clock_diff}" # fast situation, we add a plus before number
            formatted_text = "{:<10} {:>5}min\n".format(clock_label, clock_diff) # right align...
            file.write(formatted_text)


# *****------------------------------------------------
# Task 7: Finding coupling faults [5 marks]]
# *****------------------------------------------------

def check_coupling(path_1, path_2):
    """
    This function is designed to determine whether the clock is running fast or slow. It mainly
    consists of the following steps:

    1. Read the clock and calculate the actual time passed based on the difference in the 
    hour hand's position. 

    2. Use the change in error to determine whether the clock is running fast or slow.
    
    3. Calculate the change per hour and output
    """

    #-----------------------------------------------------------
    # *Step1: read the clock and get the real pass time 
    clock_array1 = read_image(path_1) # read into numpy array
    clock_array2 = read_image(path_2)
    hour_hand1, minute_hand1 = get_clock_hands(clock_array1) # get the croods of clock hands
    hour_hand2, minute_hand2 = get_clock_hands(clock_array2)
    
    hour_angle1 = get_angle(hour_hand1) # get the angle of hour hands to 12 o'clock
    hour_angle2 = get_angle(hour_hand2)
    real_angle = hour_angle2 - hour_angle1 # get the angle of real pass time 

    # Because the time might pass 12 o'clock, we cannot  simply subtract the new hour hand's angle 
    # from the old hour hand's angle as the true pass time.
    if real_angle < 0: # when pass the 12 o'clock

        # Add the small angle of the old hour hand to 12 o'clock to the angle of the new hour hand.
        real_angle = (2 * np.pi - hour_angle1) + hour_angle2
    else: 
        next

    # Calculate the real time passed based on the angle
    real_passtime_radian = real_angle / (np.pi/6)
    real_pass_hour = int(real_passtime_radian)
    real_pass_minute = (real_passtime_radian - real_pass_hour) * 60
    real_passtime = real_pass_hour * 60 + real_pass_minute

    #-----------------------------------------------------------
    # **Step2: Calculate the change in misalignment
    minute_angle1 = get_angle(minute_hand1)
    self_error1 = check_alignment(hour_angle1, minute_angle1) 
    minute_angle2 = get_angle(minute_hand2)
    self_error2 = check_alignment(hour_angle2, minute_angle2)
    self_diff = self_error2 - self_error1

    #-----------------------------------------------------------
    # ***Step3: Output text
    if self_diff == 0:
        return f"The hour and minute hand are coupled properly."
    else:
        # real_passtime is in minutes, we convert it to hours and then calculate
        diff_per_hour = int(self_diff)/real_passtime/60
        minute_diff = int(diff_per_hour)
        second_diff = round(diff_per_hour% 1 * 60)
        
        # Texting...
        if diff_per_hour < 0:
            return f"The minute hand loses {abs(minute_diff)} minutes, {abs(second_diff)} seconds per hour."
        if diff_per_hour > 0:
            return f"The minute hand gains {abs(minute_diff)} minutes, {abs(second_diff)} seconds per hour."