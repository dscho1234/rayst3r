import inspect
import os 

def gen_key(raw_key):
    # concat the raw_key with the file name that this function is called from
    current_frame = inspect.currentframe()
    # Get the caller's frame (the frame that called this function)
    caller_frame = current_frame.f_back
    # Extract the filename from the caller's frame
    caller_file = caller_frame.f_code.co_filename
    caller_file = os.path.basename(caller_file).replace(".py","")
    return f"{caller_file}_{raw_key}"