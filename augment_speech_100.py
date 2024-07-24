import os
import numpy as np
import argparse
from multiprocessing import Pool
import random
import soundfile as sf
import scipy.signal as ssi
from tqdm import tqdm
import utility
import pickle

def augment_data(speech_path, output_path, irfile_path):


    speech, fs_s = sf.read(speech_path)

    speech_length = speech.shape[0]

    ir_length_fix = 4000

    # if(speech_length>96000):
    #     speech = speech[0:96000]
    #     # sf.write(process_full_path,IR,fs_s)
    # else:
    #     zeros_len = 96000 - speech_length
    #     zeros_lis = np.zeros(zeros_len)
    #     speech = np.concatenate([speech,zeros_lis])

    # # if len(speech.shape) != 1:
    # #     speech = speech[:, 0]


    if np.issubdtype(speech.dtype, np.integer):
        speech = utility.pcm2float(speech, 'float32')    
    # convolution
    IR, fs_i = sf.read(irfile_path)

    IR_length = IR.shape[0]
    
    if(IR_length>ir_length_fix):
        IR = IR[0:ir_length_fix]
    else:
        zeros_len = ir_length_fix - IR_length
        zeros_lis = np.zeros(zeros_len)
        IR = np.concatenate([IR,zeros_lis])

    if np.issubdtype(IR.dtype, np.integer):
        IR = utility.pcm2float(IR, 'float32')
    

    # eliminate delays due to direct path propagation
    direct_idx = np.argmax(np.fabs(IR))
  
    temp = utility.smart_convolve(speech, IR)
        
    speech = np.array(temp) * 0.01

    maxval = np.max(np.fabs(speech))
    if maxval == 0:
        print("file {} not saved due to zero strength".format(speech_path))
        return -1
    if maxval >= 1:
        amp_ratio = 0.99 / maxval
        speech = speech * amp_ratio

    sf.write(output_path, speech, fs_s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='augment',
                                     description="""Script to augment dataset""")
    parser.add_argument("--pickle", "-p", default=None, help="pickle files path", type=str)
    parser.add_argument("--seed", "-s", default=0, help="Random seed", type=int)
    parser.add_argument("--nthreads", "-n", type=int, default=16, help="Number of threads to use")

    args = parser.parse_args()
    pickle_path = args.pickle
    nthreads = args.nthreads

    with open(pickle_path, 'rb') as f:
        pickle_list = pickle.load(f)

    
     

    add_reverb = True

   

    pbar = tqdm(total=len(pickle_list))
    def update(*a):
        pbar.update()
    try:
        # # Create a pool to communicate with the worker threads
        pool = Pool(processes=nthreads)
        for full_dict in pickle_list:
            ir_sample =   full_dict["mono_rir_path"]
            output_path =  full_dict["reverb_speech_path"].replace(".wav","_100.wav")
            speech_path =full_dict["clean_speech_path"]

            pool.apply_async(augment_data, args=(speech_path, output_path, ir_sample), callback=update)
    except Exception as e:
        print(str(e))
        pool.close()
    pool.close()
    pool.join() 


