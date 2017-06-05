#!/usr/bin/env python
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import csv 
import pandas as pd
from PIL import Image

SAMPLE_FILE = './AMZN/AMZN_TEST/AMZN-test.csv'
WINDOW_SIZE = 30

with open(SAMPLE_FILE,'r') as file:
    time = []
    close_price = []
    total_data = []
    earning = []

    reader = csv.reader(file)
    for line in reader:
        time.append(line[0])
        close_price.append(line[4])
    
    close_price_new = [float(str) for str in close_price]
    data_obj = pd.DataFrame(close_price,index=time,columns=['close_price'])
    data_obj = data_obj.astype(float)
 #  data_obj = data_obj.sort_index()
    for i in range(reader.line_num):
        if(i>=WINDOW_SIZE-1):
            if(i<reader.line_num-1):
                earning.append(close_price_new[i+1]-close_price_new[i])
            part_data_obj = data_obj.iloc[i-(WINDOW_SIZE-1):i,:]
            part_data_obj.plot()
            plt.savefig("./AMZN/AMZN_TEST/AMZN_PIC/{}.jpg".format(i-(WINDOW_SIZE-1)))
    
    np.savetxt("./AMZN/AMZN_TEST/earning.csv",earning,delimiter='')

