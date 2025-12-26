import tkinter as tk
from PIL import Image, ImageDraw, ImageOps, ImageTk
import numpy as np
from matplotlib import pyplot as plt

from src.model.activation_function import Softmax
from src.model.network import NN
from src.utils.image_classification import plot_image_with_colored_label


class MNISTDrawer:
    def __init__(self, trained_nn: NN):
        self.trained_nn = trained_nn

        self.canvas_size = 400      # big canvas for drawing
        self.scale = 10
        self.thumb_size = 40        # network input size

        self.img = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.img)

        # Tkinter setup
        self.root = tk.Tk()
        self.root.title("Draw a digit")

        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.canvas.pack(side="left")
        self.canvas.bind("<B1-Motion>", self.paint)

        # Right panel: thumbnail + button
        right_frame = tk.Frame(self.root)
        right_frame.pack(side="right", fill="both", expand=True)

        self.preview_label = tk.Label(right_frame)
        self.preview_label.pack(pady=10)

        # create a 10 class bar chart for prediction
        self.predict_label = tk.Label(right_frame, text="Predicted Digit: ", font=("Arial", 16))
        self.predict_label.pack(pady=10)
        self.predict_canvas = tk.Canvas(right_frame, width=200, height=200)
        self.predict_canvas.pack()

        self.clear_button = tk.Button(right_frame, text="Clear", command=self.clear)
        self.clear_button.pack(pady=10)

        self.update_preview()

    def paint(self, event):
        x, y = event.x, event.y
        r = 13
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
        self.update_preview()

    def update_preview(self):
        # Downscale to 28x28
        img_small_inv = self.img.resize((self.thumb_size, self.thumb_size), Image.Resampling.LANCZOS)
        # Invert for MNIST style
        #img_small_inv = ImageOps.invert(img_small)
        # Upscale back to show clearly
        img_preview = img_small_inv.resize((self.thumb_size*5, self.thumb_size*5), Image.Resampling.NEAREST)
        tk_img = ImageTk.PhotoImage(img_preview)
        self.preview_label.config(image=tk_img)
        self.preview_label.image = tk_img  # keep reference

        # Predict digit
        img_array = np.array(img_small_inv).astype(np.float32)
        output = self.trained_nn(img_array.reshape(1, -1))
        predicted_digit = np.argmax(output, axis=1)[0]
        self.predict_label.config(text=f"Predicted Digit: {predicted_digit}")

        # Plot prediction probabilities
        self.predict_canvas.delete("all")
        probs = Softmax().forward(output)
        probs = probs.flatten()
        bar_width = 15
        max_height = 150
        for i in range(10):
            bar_height = probs[i] * max_height
            x0 = 10 + i * (bar_width + 5)
            y0 = 180 - bar_height
            x1 = x0 + bar_width
            y1 = 180
            self.predict_canvas.create_rectangle(x0, y0, x1, y1, fill="blue")
            self.predict_canvas.create_text(x0 + bar_width/2, 190, text=str(i))

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill=0)
        self.update_preview()

    def run(self):
        self.root.mainloop()
        return np.array(self.img.resize((self.thumb_size, self.thumb_size), Image.Resampling.LANCZOS))

if __name__ == "__main__":
    drawer = MNISTDrawer(None)
    digit_array = drawer.run()
    print("Shape:", digit_array.shape)