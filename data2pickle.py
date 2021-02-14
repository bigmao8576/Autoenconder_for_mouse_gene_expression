
"""
If we want to try k fold cross validation, we need to write all the data into pickles

Here, we hope to save all the fold info in the hard drive. 

We can also keep the fold info in memory, however, we found it is a little bit risky,
in case we got core dumped, hardware crash or even auto reboot.


Warning, if you run this file, the original pickle file(fold_dict_0601.pkl) will be overwritten.
"""

import pickle
import utils


'''
If you hope to use the data listed here, please cite the following work:
    
@article{pan202012,
  title={12-h clock regulation of genetic information flow by XBP1s},
  author={Pan, Yinghong and Ballance, Heather and Meng, Huan and Gonzalez, Naomi and Kim, Sam-Moon and Abdurehman, Leymaan and York, Brian and Chen, Xi and Schnytzer, Yisrael and Levy, Oren and others},
  journal={PLoS biology},
  volume={18},
  number={1},
  pages={e3000580},
  year={2020},
  publisher={Public Library of Science San Francisco, CA USA}
}

'''

path =  'data_rhythm_20200601.csv'


#the first dataset is not currently available, but it will be released in recent future
#path = 'dataProcessed.csv'

if path == 'data_rhythm_20200601.csv':
    save_path = 'fold_dict_0601.pkl'
    channel_num=2

else:
    save_path = 'fold_dict_1030.pkl'
    channel_num=3

data_dict,sample_len = utils.file2dict(path,channel_num=channel_num,norm=True)

data_fold,fold_name = utils.data2fold(data_dict,5)

print('Warning,the original pickle file(fold_dict_0601.pkl) will be overwritten')

pickle.dump(data_fold,open(save_path,'wb'),protocol=2)


