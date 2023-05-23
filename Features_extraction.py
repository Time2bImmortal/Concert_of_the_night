import os


def list_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        return files
    except FileNotFoundError:
        print("The specified directory does not exist.")
        return []
    except Exception as e:
        print("An error occurred while accessing the directory.")
        print(str(e))
        return []

