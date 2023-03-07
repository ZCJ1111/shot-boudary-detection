# optical-flow-transitions

The main file to be run in here is main-end-to-end.py

This takes several arguments as follows:

filename = the path to the mp4 which you would like to run through transition detection algorithm: str
output_path0_model = the path to the folder which contains the trained CNN model: str
output_path0 = the path to write all results to: str
hist_eq0 = whether to use histogram equalisation in the CNN (should be 1 usually):  0/1
cleanup = whether to delete all intermediate data during the run (usually 1): 0/1
start_frame = the frame of the mp4 to start processing at: int
end_frame = the final frame to be processed: int
seek_forward = how many frames to "look" forward for transitions (usually 1 except for smooth fades): int

Example:

python3 main-end-to-end.py data/example.mp4 model/ output_files/ True False 0 100 5
"# shot-boudary-detection" 
