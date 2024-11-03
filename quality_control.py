# Define all functions in this module.
def read_image(path, show=False):
    import matplotlib.pyplot as plt
    import numpy as np
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

def get_clock_hands(clock_RGB):
    import numpy as np
    gap = 0.15
    hour_hand = []
    minute_hand = []
    for x in range(clock_RGB.shape[0]):
        for y in range(clock_RGB.shape[1]):
            vec = clock_RGB[x,y]
            index = [x,y]
            if vec[0] >= vec[1] + gap and vec[0] >= vec[2] + gap:
                hour_hand.append(index)
            elif vec[1] >= vec[0] + gap and vec[1] >= vec[2] + gap:
                minute_hand.append(index)
    
    return hour_hand, minute_hand

def get_angle(coords):
    import numpy as np

    if isinstance(coords, np.ndarray):
        True
    else:
        coords = np.array(coords)
    pixels_x = coords[:, 0]
    pixels_y = coords[:, 1]

    if len(np.unique(pixels_x)) in [1, 2]:
        if max(pixels_y) < 49:
            radian = 3*np.pi/2
        else:
            radian = np.pi/2

    elif len(np.unique(pixels_y)) in [1, 2]:
        if max(pixels_x) < 49:
            radian = 0
        else:
            radian = np.pi

    else:
        theta, inter = np.polyfit(pixels_x, pixels_y, 1)
        greater_than_51 = pixels_x >= 51
        x_count = np.sum(greater_than_51)
        
        if x_count >= 5:
            radian = np.pi - np.arctan(theta)
        else:
            if np.arctan(theta) < 0:
                radian = abs(np.arctan(theta))
            else: 
                radian = 2*np.pi - np.arctan(theta)
    
    return radian
def analog_to_digital(angle_hour, angle_minute):
    import numpy as np
    angle_hour,angle_minute = float(angle_hour),float(angle_minute)
    time_hour = int((angle_hour / (np.pi/6)) //1)
    time_minute = round(angle_minute / (np.pi/6) * 5)

    if time_minute == 60:
        time_hour =+ 1 
    else:
        next

    if time_hour == 0:
        return f"12:{time_minute}"
    if time_hour//10 == 0:
        return f"0{time_hour}:{time_minute}"
    else:
        return f"{time_hour}:{time_minute}"

def check_alignment(angle_hour, angle_minute):
    import numpy as np
    angle_hour,angle_minute = float(angle_hour),float(angle_minute)
    correct_hour = int((angle_hour / (np.pi/6)) //1)
    correct_minute = round(((angle_hour / (np.pi/6)) - correct_hour)*5)
    correct_total = 60*correct_hour + correct_minute

    wrong_hour = int((angle_hour / (np.pi/6)) //1)
    wrong_minute = round(angle_minute / (np.pi/6) * 5)
    wrong_total = 60*wrong_hour + wrong_minute

    diff = correct_total - wrong_total
    if abs(diff) > 15:#!!!不确定是否正确
        return -abs(60-diff)
    else:
        return diff


def validate_batch(folder_path, tolerance):
    from datetime import datetime
    from pathlib import Path
    import os
    import numpy as np

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
        hour_hand, minute_hand = get_clock_hands(clock_array)

        hour_angle = get_angle(hour_hand)
        minute_angle = get_angle(minute_hand)

        diff = check_alignment(hour_angle, minute_angle)
        if abs(diff) > tolerance:
            abnormal_sum += 1
            abnormal_list.append(clock)
            abnormal_diffs.append(diff)
    rate_batch = round((1 - (abnormal_sum/check_sum)) * 100, 1)
    
    abnormal_dict = dict(zip(abnormal_list, abnormal_diffs))
    sorted_abnormal = dict(sorted(abnormal_dict.items(), key=lambda item: abs(item[1]), reverse=True))

    lines = [
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
        file.writelines(lines)
        for clock_label, clock_diff in abnormal_dict.items():
            if clock_diff > 0:
                file.write(f"{clock_label}: +{clock_diff:>5}min\n")
            else:
                file.write(f"{clock_label}: {clock_diff:>5}min\n")

