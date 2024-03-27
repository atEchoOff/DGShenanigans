import matplotlib.pyplot as plt
import imageio
from io import BytesIO
from IPython.display import Image

def animate(func):
    def wrapper():
        # Animate the given function, which yields whenever ready to plot
        # Create an empty list to store the frames
        frames = []

        for _ in func():
            # Convert plot to image and add to frames
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            plt.close()

        # Save frames as a GIF in memory
        gif_buf = BytesIO()
        imageio.mimsave(gif_buf, frames, 'GIF', loop=0)
        gif_buf.seek(0)

        # Display the GIF
        return Image(data=gif_buf.read(), format='gif')
    
    return wrapper