import os
import shutil

def move_files_to_folders(source_folder, destination_folder_base, train_list, val_list, test_list):
    # Ensure the source folder exists
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' does not exist.")
        return

    # Create destination folders if they don't exist
    for folder_name in ["train", "val", "test"]:
        folder_path = os.path.join(destination_folder_base, folder_name)
        os.makedirs(folder_path, exist_ok=True)

    # Get the list of txt files in the source folder
    txt_files = [file for file in os.listdir(source_folder) if file.endswith(".npy")]

    # Distribute files to destination folders based on predefined lists
    for filename in txt_files:
        file_number = filename.split(".")[0].split("_")[1]


        if file_number in train_list:
            destination_folder = "train"
        elif file_number in val_list:
            destination_folder = "val"
        elif file_number in test_list:
            destination_folder = "test"
        else:
            print(f"Ignoring file '{filename}' as it's not in train, val, or test list.")
            continue

        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder_base, destination_folder, filename)
        shutil.copy(source_path, destination_path)
        print(f"Moved '{filename}' to '{destination_folder}'.")

# Define paths and lists
source_folder_path = "D:/Phdworks/data/vocation/ant_processed"
destination_folder_base_path = "D:/Phdworks/data/vocation/annotation"

train_list=['1', '10', '100', '101', '102', '103', '104', '105', '106', '107', '108',
            '109', '11', '110', '111', '112', '113', '114', '115', '116', '117', '118',
            '119', '12', '120', '121', '122', '123', '124', '125', '126', '127', '128',
            '129', '13', '130', '131', '132', '133', '134', '135', '136', '137', '138',
            '139', '14', '140', '141', '142', '143', '144', '145', '146', '147', '148',
            '149', '15', '150', '151', '152', '153', '154', '155', '156', '157', '158',
            '159', '16', '160', '161', '162', '163', '164', '165', '166', '167', '168',
            '169', '17', '170', '171', '172', '173', '174', '175', '176', '177', '178',
            '179', '18', '180', '181', '182', '183', '184', '185', '186', '187', '188',
            '189', '19', '190', '191', '192', '193', '194', '195', '196', '197', '198',
            '199', '2', '20', '200', '201', '202', '203', '204', '205', '206', '207',
            '208', '209', '21', '210', '211', '212', '213', '214', '215', '216', '218',
            '219', '22', '220', '221', '223', '224', '225', '228', '229', '23', '231',
            '234', '235', '237', '24', '25', '26', '27', '28', '29', '30', '35', '37',
            '4', '43', '47', '50', '59', '66', '67', '68', '69']

val_list=['222', '226', '227', '3', '31', '32', '33', '34', '36', '38', '39', '40',
          '41', '42', '44', '45', '48', '49', '5', '51', '52', '53', '54', '55', '56',
          '57', '6', '60', '61', '70', '72', '73']

test_list=['217','230', '232', '233', '236', '46', '58', '62', '63', '64', '65', '7', '71',
           '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85',
           '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']
# Move files to folders
move_files_to_folders(source_folder_path, destination_folder_base_path, train_list, val_list, test_list)