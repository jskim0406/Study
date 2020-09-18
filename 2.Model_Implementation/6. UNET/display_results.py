import os
import numpy as np
import matplotlib.pyplot as plt

##
result_dir = './drive/My Drive/Colab Notebooks/6. Unet/pytorch_unet_36/results/numpy'

lst_data = os.listdir(result_dir)

lst_label = [lst for lst in lst_data if lst.startswith('label')]
lst_input = [lst for lst in lst_data if lst.startswith('input')]
lst_output = [lst for lst in lst_data if lst.startswith('output')]

lst_label.sort()
lst_input.sort()
lst_output.sort()

##
id = 0

label = np.load(os.path.join(result_dir, lst_label[id]))
input = np.load(os.path.join(result_dir, lst_input[id]))
output = np.load(os.path.join(result_dir, lst_output[id]))

##
plt.subplot(131)
plt.imshow(label, cmap='gray')
plt.title("label")

plt.subplot(132)
plt.imshow(input, cmap='gray')
plt.title("input")

plt.subplot(133)
plt.imshow(output, cmap='gray')
plt.title("output")
plt.show()



