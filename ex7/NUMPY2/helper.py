import matplotlib.pyplot as plt
import numpy as np

def load_image(file_path):
    img = plt.imread(file_path)
    return img

# Define a function to display an image
## "plt.tick_params( )" is used to change the appearance of ticks, tick labels, and gridlines.
### Use "imshow()" function of pyplot to display image

def show_image(img):
    plt.tick_params(axis = 'both',       # changes apply to both x-axis and y-axis
                  which ='both',       # both major and minor ticks are affected
                  left = False,        # ticks along the left edge are off
                  right = False,       # ticks along the right edge are off
                  bottom = False,      # ticks along the bottom edge are off
                  top = False,         # ticks along the top edge are off
                  labelbottom = False, # bottom lable is off
                  labelleft = False)   # left lable is off
                  
    plt.imshow(img) 
    plt.show()