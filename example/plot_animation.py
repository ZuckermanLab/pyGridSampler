import os
import imageio
from natsort import natsorted


# Directory containing the images
dir_path = '/Users/georgeau/Desktop/GitHub/GridSampler/'

# List of image file names, sorted numerically
image_files = natsorted([f for f in os.listdir(dir_path) if f.endswith('.png')], key=lambda x: int(x.split('_')[1].split('.')[0]))

# List of image file names for the 'test_' images, sorted numerically
test_files = natsorted([f for f in os.listdir(dir_path) if f.startswith('testrun') and f.endswith('.png')], key=lambda x: int(x.split('_')[1].split('.')[0]))


# Output file names for the GIF animations
output_file_test = 'animation_test.gif'


# Create the 'test_' GIF animation
duration = 0.042 * 2
images_test = []
for f in test_files:
    file_path = os.path.join(dir_path, f)
    images_test.append(imageio.imread(file_path))
imageio.mimsave(output_file_test, images_test, duration=duration)


