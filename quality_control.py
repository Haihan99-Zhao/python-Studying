# Define all functions in this module.
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import os

def read_image(path, show=False):
    image = plt.imread(path)

    if show :
        fig, ax = plt.subplots(1, image.shape[2]+1, figsize=(10,8))
        ax[0].imshow(image)
        ax[0].set_title("The original plot")
        ax[0].axis('off')
        color_set = ["Red", "Green", "Blue", "Alpha"]
        for i in range(image.shape[2]):
            tmpt_image = np.zeros_like(image)
            tmpt_image[:, :, i] = image[:, :, i] 
            ax[i+1].imshow(tmpt_image)
            ax[i+1].set_title(f"The {color_set[i]} plot")
            ax[i+1].axis('off')
        plt.show()
        if image.shape[-1] == 3:
            print(f' The format of file is RGB.\n')
        else:
            print(f' The format of file is RGBA.\n')

    return image

def outliers_clean_points(clock_RGB, n=3):

    each_std = np.std(clock_RGB, axis=2)
    reshaped_std = each_std.reshape(-1)
    total_std = np.std(reshaped_std)
    total_mean = np.mean(reshaped_std)
    upper_bound = total_mean + n * total_std

    new_image = np.ones_like(clock_RGB)

    for i in range(clock_RGB.shape[0]):
        for j in range(clock_RGB.shape[1]):
            pixel = clock_RGB[i, j]

            pixel_variance = np.std(pixel)
    

            if pixel_variance >= upper_bound:
                new_image[i,j] = pixel

    return new_image


def get_clock_hands(clock_RGB):

   
    hour_hand = []
    minute_hand = []
    gap = 0.1
    for x in range(clock_RGB.shape[0]):
        for y in range(clock_RGB.shape[1]):
            vec = clock_RGB[x,y]
            index = [x,y]
            if vec[0] > vec[1] + gap  and vec[0] > vec[2] + gap:
                hour_hand.append(index)
            elif vec[1] > vec[0] + gap and vec[1] > vec[2] + gap:
                minute_hand.append(index)
    
    return hour_hand, minute_hand

def get_angle(coords):

    if isinstance(coords, np.ndarray):
        True
    else:
        coords = np.array(coords)

    pixels_x = coords[:, 0]
    pixels_y = coords[:, 1]

    if len(np.unique(pixels_x)) == 1:
        if np.median(pixels_y) < 49:
            radian = 3 * np.pi / 2
        else:
            radian = np.pi / 2

    elif len(np.unique(pixels_y)) ==1 :
        if np.median(pixels_x) < 49:
            radian = 0
        else:
            radian = np.pi

    else:
        slope, inter = np.polyfit(pixels_x, pixels_y, 1)
        # if abs(slope) >= 1:
        #     slope = round(slope,2)
        # elif abs(slope) >= 10:
        #     slope = slope
        # else:
        #     slope = round(slope,2)
        if np.median(pixels_x) >= 51:
            if slope == np.inf and np.median(pixels_y) > 50:
                radian = np.pi / 2
            elif slope == np.inf and np.median(pixels_y) <= 50:
                radian = 3*np.pi / 2
            else:
                radian = np.pi - np.arctan(slope)
        else:
            if np.median(pixels_y) >= 51 or np.arctan(slope) < 0:
                radian = - np.arctan(slope)
            else: 
                radian = 2*np.pi - np.arctan(slope)
    
    return round(radian,1)

def analog_to_digital(angle_hour, angle_minute):
    
    angle_hour,angle_minute = float(angle_hour),float(angle_minute)
    time_hour = int(angle_hour / (np.pi/6))
    time_minute = int(angle_minute / (np.pi/6) * 5)
    
    if time_hour == 0:
        time_hour = 12
    modified_hour = f"0{time_hour}"
    modified_minute = f"0{time_minute}"
    
    return f"{modified_hour[-2:]}:{modified_minute[-2:]}"

def check_alignment(angle_hour, angle_minute):
    
    angle_hour,angle_minute = float(angle_hour),float(angle_minute)
    correct_hour = angle_hour // (np.pi/6)

    correct_minute = ((angle_hour / (np.pi/6)) - correct_hour) * 60
    wrong_minute = angle_minute / (np.pi/6) * 5
    
    diff = wrong_minute - correct_minute
    if abs(diff) > 30:
        if diff <0:
            return diff+60
        else:
            return diff-60
    else:
        return diff


def validate_batch(folder_path, tolerance):
    

    number_batch = folder_path[-1]

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d, %H:%M:%S")

    folder = Path(folder_path)
    files = list(folder.glob("*.png"))
    check_sum = len(files)

    abnormal_sum = 0
    abnormal_list = []
    abnormal_diffs = []
    
    for clock in files:
        clock_array = read_image(clock)
        new_array = outliers_clean_points(clock_array)
        hour_hand, minute_hand = get_clock_hands(new_array)

        hour_angle = get_angle(hour_hand)
        minute_angle = get_angle(minute_hand)

        diff = int(check_alignment(hour_angle, minute_angle))
        if abs(diff) > tolerance:
            abnormal_name = clock.stem 
            abnormal_sum += 1
            abnormal_list.append(abnormal_name)
            abnormal_diffs.append(diff)
        
    rate_batch = round((1 - (abnormal_sum/check_sum)) * 100, 1)
    
    abnormal_dict = dict(zip(abnormal_list, abnormal_diffs))
    sorted_abnormal = dict(sorted(abnormal_dict.items(), key=lambda item: abs(item[1]), reverse=True))

    report_main = [
        f"Batch number:{number_batch}\n"
        f"Checked on {formatted_datetime}\n"
        "\n"
        f"Total number of clocks:{check_sum} "
        "\n"
        f"Number of clocks passing quality control ({tolerance}-minute tolerance): {check_sum - abnormal_sum}"
        "\n"
        f"Batch quality: {rate_batch}%"
        "\n"
        "Clocks to send back for readjustment:"
        "\n"
    ]
    if not os.path.exists("QC_reports"):
        os.makedirs("QC_reports")

    report_file_path = os.path.join("QC_reports", f'batch_{number_batch}_QC.txt')

    with open(report_file_path, 'w') as file:
        file.writelines(report_main)
        for clock_label, clock_diff in sorted_abnormal.items():
            if clock_diff > 0:
                clock_diff = f"+{clock_diff}"
            formatted_text = "{:<10} {:>5}min\n".format(clock_label, clock_diff)
            file.write(formatted_text)



def real_time(angle_hour):
    hour = angle_hour // (np.pi/6)
    minute = (angle_hour / (np.pi/6) - hour) * 60
    # minute = angle_minute / (np.pi/6) * 5
    total_time = hour * 60 + minute

    return round(total_time,3)

def check_coupling(path_1, path_2):
    clock_array1 = read_image(path_1)
    clock_array2 = read_image(path_2)

    new_array1 = outliers_clean_points(clock_array1)
    new_array2 = outliers_clean_points(clock_array2)

    hour_hand1, minute_hand1 = get_clock_hands(new_array1)
    hour_hand2, minute_hand2 = get_clock_hands(new_array2)
    
    hour_angle1 = get_angle(hour_hand1)
    minute_angle1 = get_angle(minute_hand1)

    real_time1 = real_time(hour_angle1)
    #clock_time1 = analog_to_digital(hour_angle1, minute_angle1)
    self_error1 = check_alignment(hour_angle1, minute_angle1) 

    hour_angle2 = get_angle(hour_hand2)
    minute_angle2 = get_angle(minute_hand2)

    real_time2 = real_time(hour_angle2)
    #clock_time2 = real_time(hour_angle2, minute_angle2)
    self_error2 = check_alignment(hour_angle2, minute_angle2)

    #real_time1 = clock_time1 - self_error1
    #real_time2 = clock_time2 - self_error2
    #print(clock_time1)
    self_diff = int(self_error2) - int(self_error1)
    real_diff = int(round(real_time2,2) - round(real_time1,2))
    print(self_error1)
    print(self_error2)
    print(real_time1)
    print(real_time2)
    print(self_diff)
    print(real_diff)
    if self_diff == 0:
        return f"The hour and minute hand are coupled properly."
    else:
        diff_per_hour = self_diff/(real_diff/60)
        minute_diff = int(diff_per_hour)
        second_diff = round(diff_per_hour % 1 * 60)
        
        if diff_per_hour < 0:
            return f"The minute hand loses {abs(minute_diff)} minutes, {abs(second_diff)} seconds per hour."
        if diff_per_hour > 0:
            return f"The minute hand gains {abs(minute_diff)} minutes, {abs(second_diff)} seconds per hour."