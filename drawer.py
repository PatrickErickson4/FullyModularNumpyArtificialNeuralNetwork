'''
@author: Patrick Erickson, GPT-o3-mini-high
Project Name: DrawingCanves
Project Description: Creates a window for users to draw numbers. My models will try to guess your number!
This class was heavily assisted by GPT-o3-mini-high.
'''

from NeuralNetworkScripts.NeuralNetwork import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import messagebox

class DrawingCanvas:
    '''
    Creates a interactive drawing popout to test efficacies of models.
    Takes in user input.
    NOTE: It probably won't be that good, since the model generalizes too much
    with respect to how little variation there is in MNIST.
    '''
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.setup_canvas(title="Let me Guess your number! \n(Be nice,I have no eyes)\nPress Enter to input guesses and e to clear if you mess up.")
        
        # Data for drawing (using NaN values to separate strokes)
        self.xdata, self.ydata = [], []
        self.line, = self.ax.plot([], [], lw=16, color='black')
        self.is_drawing = False

        # Container for processed matrices
        self.matrices = []  # List to store each 28x28 matrix

        # Connect mouse and keyboard events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def setup_canvas(self, title):
        self.ax.cla()  # Clear the axes
        self.ax.set_title(title)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')  # Hide axes for a clean canvas

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.is_drawing = True
        # Insert a break if needed to avoid connecting lines between strokes
        if self.xdata and not np.isnan(self.xdata[-1]):
            self.xdata.append(np.nan)
            self.ydata.append(np.nan)
        self.xdata.append(event.xdata)
        self.ydata.append(event.ydata)
        self.line.set_data(self.xdata, self.ydata)
        self.fig.canvas.draw()

    def on_motion(self, event):
        if not self.is_drawing or event.inaxes != self.ax:
            return
        self.xdata.append(event.xdata)
        self.ydata.append(event.ydata)
        self.line.set_data(self.xdata, self.ydata)
        self.fig.canvas.draw()

    def on_release(self, event):
        if self.is_drawing:
            self.is_drawing = False
            # End the current stroke with a break
            self.xdata.append(np.nan)
            self.ydata.append(np.nan)
            self.line.set_data(self.xdata, self.ydata)
            self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'e':
            # Clear the entire canvas immediately when 'e' is pressed
            self.clear_canvas()
        elif event.key in ['enter', 'return']:
            # Process the drawing to a 28x28 matrix, save it, then clear the canvas
            matrix = self.convert_drawing_to_matrix()
            self.matrices.append(matrix)
            self.clear_canvas()

    def convert_drawing_to_matrix(self):
        # Ensure the canvas is up to date
        self.fig.canvas.draw()

        # Get canvas dimensions
        width, height = self.fig.canvas.get_width_height()

        # Extract the canvas content as an RGBA array, then discard the alpha channel
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        rgb_array = buf[:, :, :3]
        pil_img = Image.fromarray(rgb_array)

        # Convert to grayscale ('L' mode) and resize to 28x28 pixels using LANCZOS for quality
        gray_img = pil_img.convert('L')
        resized_img = gray_img.resize((28, 28), Image.LANCZOS)
        matrix = np.array(resized_img)

        # Invert colors: drawn parts (originally black) become 255, background becomes 0
        matrix = 255 - matrix

        return matrix

    def clear_canvas(self):
        # Reset drawing data and clear the canvas
        self.xdata = []
        self.ydata = []
        self.setup_canvas(title="Let me Guess your number! \n(Be nice,I have no eyes)\nPress Enter to input guesses and e to clear if you mess up.")
        self.line, = self.ax.plot([], [], lw=16, color='black')
        self.fig.canvas.draw()


if __name__ == '__main__':
    canvas = DrawingCanvas()
    plt.show()  # This call blocks until the window is closed
    
    # After closing the window, convert the saved matrices into a tensor
    numbers = []
    if canvas.matrices:
        numbers = np.stack(canvas.matrices)  # Shape will be (n, 28, 28)
    else:
        print("No matrices were saved.")
    
    if len(numbers) != 0:
        numbers = numbers.reshape(len(numbers), 28*28)
        savedModel = NeuralNetwork(model="InteractivesAndModels/MNISTModelRandom")
        guessesSaved = []
        for i, item in enumerate(numbers):
            toAdd = numbers[i:i+1, :]
            guessesSaved.append(savedModel.predict(toAdd))
        guesses = [int(np.argmax(arr)) for arr in guessesSaved]
        
        # Limit to 100 guesses if more than 100 are produced
        if len(guesses) > 100:
            guesses = guesses[:100]
        
        # Create a comma-separated string of guesses
        guesses_str = ', '.join(str(g) for g in guesses)
        
        # Create a Tkinter pop-up message box to display the guesses
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        messagebox.showinfo("Guesses", guesses_str)