import matplotlib.pyplot as plt
import numpy as np

def plot_img_and_mask(img, mask):
    
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    
    ax[0].set_title('Input image')
    
    ax[0].imshow(img)
    
    for i in range(classes):
        
        copy_mask = mask.copy()
        
        ax[i + 1].set_title(f'Mask (class {i})')
        
        result = np.where(copy_mask, 1, 255)

        ax[i + 1].imshow(result)
        
    plt.xticks([]), plt.yticks([])
    
    plt.show()
    
