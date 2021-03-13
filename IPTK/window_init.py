import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import skimage.io as io
from neighborhood_processing import box_smoothing, gaussian_smoothing, laplacian_sharping, order_statistic, \
    highboost
from point_processing import negative, intensity_level_slicing, contrast_stretching, power_law, histogram, \
    global_histogram_equalization, local_histogram_equalization
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# this function shows result of histogram
def histogram_invoker(file_path):
    result = histogram(file_path)
    if result == -1:
        messagebox.showerror('Unsupported image', 'Please import the right format image')
        return


# resize pic to most suitable size into canvas
def resize(w_box, h_box, image):
    original_width, original_height = image.size
    f1 = 1.0 * w_box / original_width
    f2 = 1.0 * h_box / original_height
    factor = min([f1, f2])
    resized_width = int(original_width * factor)
    resize_height = int(original_height * factor)
    return image.resize((resized_width, resize_height), Image.ANTIALIAS)


# this function shows result of mask image in sharping
def mask_invoker(file_path):
    try:
        mask = io.imread(file_path)
        plt.imshow(mask, cmap=plt.cm.gray, vmin=0, vmax=255)
        plt.title("Mask", fontsize=6)
        plt.axis('off')
        plt.show()
    except Exception:
        messagebox.showerror('Unsupported image', 'Please import the right format image')
        return


# this function handles exception
def default_invoker():
    messagebox.showerror('Empty', 'Please import the image')
    return


class Window:
    root = None  # root window
    image = None  # imported image
    file_path = None  # imported file path

    # following member fields of point processing
    negative_image = None
    bit_plane_selection = 0
    intensity_level_slicing_image = None
    stretched_image = None
    power_law_image = None
    power_law_constant = 0
    power_law_gamma = 0
    global_histogram_equalized_image = None
    local_histogram_equalized_image = None

    # following member fields of neighborhood processing
    box_kernel_size = 0
    box_smoothing_image = None
    
    gaussian_kernel_size = 0
    gaussian_constant_k = 0
    gaussian_sigma = 0
    gaussian_smoothing_image = None

    laplacian_sharping_image = None
    laplacian_sharping_mask = None

    median_filtering_image = None
    
    highboost_kernel_size = 0
    highboost_constant_k = 0
    highboost_sigma = 0
    highboost_image = None
    highboost_mask = None

    # window main components
    main_frame = None
    left_frame = None
    left_canvas = None
    left_hist_button = None
    right_frame = None
    right_canvas = None
    right_hist_button = None
    right_show_mask_button = None

    # IO image path
    GIF_PATH = 'current.gif'
    NEGATIVE_PATH = 'negative.gif'
    INTENSITY_LEVEL_SLICING_PATH = 'intensity_level_slicing.gif'
    CONTRAST_STRETCHING_PATH = 'contrast_stretching.gif'
    POWER_LAW_PATH = 'power.gif'
    GLOBAL_HISTOGRAM_EQUALIZED_PATH = 'global_histogram_equalized.gif'
    LOCAL_HISTOGRAM_EQUALIZED_PATH = 'local_histogram_equalized.gif'
    BOX_SMOOTHING_PATH = 'box_smoothing.gif'
    GAUSSIAN_SMOOTHING_PATH = 'gaussian_smoothing.gif'
    LAPLACIAN_SHARPING_PATH = 'laplacian_sharping.gif'
    LAPLACIAN_SHARPING_MASK_PATH = 'laplacian_sharping_mask.gif'
    HIGHBOOST_PATH = 'highboost.gif'
    HIGHBOOST_MASK_PATH = 'highboost_mask.gif'
    MEDIAN_FILTERING_PATH = 'median_filtering.gif'

    def __init__(self):

        self.root = tk.Tk()
        self.root.title('Image Processing Toolkit')
        self.root.geometry('700x430')
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack()
        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side='left')
        self.left_canvas = tk.Canvas(self.left_frame, height=350, width=330)
        self.left_canvas.pack()
        self.left_hist_button = tk.Button(self.left_frame, text='Show Histogram',
                                          command=lambda: histogram_invoker(file_path=self.file_path))

        self.right_frame = tk.Frame(self.main_frame)
        self.right_frame.pack(side='right')
        self.right_canvas = tk.Canvas(self.right_frame, height=350, width=330)
        self.right_canvas.pack()
        self.right_show_mask_button = tk.Button(self.right_frame, text='Show mask', command=default_invoker)
        self.right_hist_button = tk.Button(self.right_frame, text='Show Histogram', command=default_invoker)

    def create_menu(self, root):
        menuBar = tk.Menu(root)
        fileMenu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label='File', menu=fileMenu)
        fileMenu.add_command(label='Import Image', command=self.import_image)
        fileMenu.add_command(label='Exit', command=lambda: exit(0))

        pointMenu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label='Point Processing', menu=pointMenu)
        pointMenu.add_command(label='Negative', command=self.do_negative)
        pointMenu.add_command(label='Intensity-Level Slicing', command=self.do_intensity_level_slicing)
        pointMenu.add_command(label='Contrast Stretching', command=self.do_contrast_stretching)
        pointMenu.add_command(label='Global Histogram Equalization', command=self.do_global_histogram_equalization)
        pointMenu.add_command(label='Local Histogram Equalization', command=self.do_local_histogram_equalization)
        pointMenu.add_command(label='Power-Law', command=self.do_power_law)

        neighborhoodMenu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label='Neighborhood Processing', menu=neighborhoodMenu)
        smoothingMenu = tk.Menu(neighborhoodMenu)
        sharpingMenu = tk.Menu(neighborhoodMenu)
        smoothingMenu.add_command(label='Box Smoothing', command=self.do_box_smoothing)
        smoothingMenu.add_command(label='Gaussian Smoothing', command=self.do_gaussian_smoothing)
        sharpingMenu.add_command(label='Laplacian Sharping', command=self.do_laplacian_sharping)
        sharpingMenu.add_command(label='Unsharp Mask and Highboost', command=self.do_highboost)
        neighborhoodMenu.add_cascade(label='Smoothing', menu=smoothingMenu)
        neighborhoodMenu.add_cascade(label='Sharping', menu=sharpingMenu)
        neighborhoodMenu.add_command(label='Median filtering', command=self.do_median_filtering)
        root.config(menu=menuBar)

    def import_image(self):
        self.file_path = filedialog.askopenfilename()
        # resize then convert to PhotoImage
        try:
            image = Image.open(self.file_path)
            image = resize(300, 400, image)
            image.save(self.GIF_PATH)
            self.image = tk.PhotoImage(file=self.GIF_PATH)
            self.left_canvas.create_image(20, 20, anchor='nw', image=self.image)
            self.left_hist_button.pack(side='bottom')

        except ValueError:
            messagebox.showerror('Unsupported image', 'Please import the right format image')
            return
        except AttributeError:
            messagebox.showerror('Unsupported image', 'Please import the right format image')
            return

    def do_negative(self):
        self.right_show_mask_button.config(command=lambda: mask_invoker(file_path=default_invoker))
        self.right_show_mask_button.pack_forget()
        negative_arr = negative(self.GIF_PATH)
        if negative_arr is None:
            messagebox.showerror('Unsupported image', 'Please import the right format image')
            return
        Image.fromarray(np.uint8(negative_arr)).save(self.NEGATIVE_PATH)
        self.negative_image = tk.PhotoImage(file=self.NEGATIVE_PATH)
        self.right_canvas.create_image(20, 20, anchor='nw', image=self.negative_image)
        self.right_hist_button.config(command=lambda: histogram_invoker(file_path=self.NEGATIVE_PATH))
        self.right_hist_button.pack()

    def do_intensity_level_slicing(self):
        confirm_window = tk.Toplevel(self.root)
        confirm_window.geometry('300x300')
        confirm_label = tk.Label(confirm_window, text="Please select bit plane level:")
        confirm_label.pack()
        selection_list = tk.Listbox(confirm_window)
        for item in [1, 2, 3, 4, 5, 6, 7, 8]:
            selection_list.insert("end", item)
        selection_list.pack()
        selection_list.bind('<<ListboxSelect>>', self.getSelectedItem)
        confirm_button = tk.Button(confirm_window, text='Confirm', width=10, height=3,
                                   command=self.intensity_level_slicing_invoker)
        confirm_button.pack()

    # call back function for selecting intensity level
    def getSelectedItem(self, evt):
        # Note here that Tkinter passes an event object
        w = evt.widget
        index = int(w.curselection()[0])
        self.bit_plane_selection = w.get(index)

    def intensity_level_slicing_invoker(self):
        self.right_show_mask_button.config(command=lambda: mask_invoker(file_path=default_invoker))
        self.right_show_mask_button.pack_forget()
        intensity_level_slicing_arr = intensity_level_slicing(self.GIF_PATH, self.bit_plane_selection)
        if intensity_level_slicing_arr is None:
            messagebox.showerror('Unsupported image', 'Please import the right format image')
            return
        Image.fromarray(np.uint8(intensity_level_slicing_arr)).save(self.INTENSITY_LEVEL_SLICING_PATH)
        self.intensity_level_slicing_image = tk.PhotoImage(file=self.INTENSITY_LEVEL_SLICING_PATH)
        self.right_canvas.create_image(20, 20, anchor='nw', image=self.intensity_level_slicing_image)
        self.right_hist_button.config(command=lambda: histogram_invoker(file_path=self.INTENSITY_LEVEL_SLICING_PATH))
        self.right_hist_button.pack()

    def do_contrast_stretching(self):
        self.right_show_mask_button.config(command=lambda: mask_invoker(file_path=default_invoker))
        self.right_show_mask_button.pack_forget()
        stretched_arr = contrast_stretching(self.GIF_PATH)
        if stretched_arr is None:
            messagebox.showerror('Unsupported image', 'Please import the right format image')
            return
        Image.fromarray(stretched_arr).save(self.CONTRAST_STRETCHING_PATH)
        self.stretched_image = tk.PhotoImage(file=self.CONTRAST_STRETCHING_PATH)
        self.right_canvas.create_image(20, 20, anchor='nw', image=self.stretched_image)
        self.right_hist_button.config(
            command=lambda: histogram_invoker(file_path=self.CONTRAST_STRETCHING_PATH))
        self.right_hist_button.pack()

    def do_power_law(self):
        confirm_window = tk.Toplevel(self.root)
        confirm_window.geometry('300x150')
        constant_label = tk.Label(confirm_window, text="Please input constant value:")
        constant_label.pack()
        constant_entry = tk.Entry(confirm_window, show=None)
        constant_entry.bind('<KeyRelease>', self.power_law_constant_assign)
        constant_entry.pack()
        gamma_label = tk.Label(confirm_window, text="Please input gamma value:")
        gamma_label.pack()
        gamma_entry = tk.Entry(confirm_window, show=None)
        gamma_entry.bind('<KeyRelease>', self.gamma_assign)
        gamma_entry.pack()
        confirm_button = tk.Button(confirm_window, text='Confirm', width=10, height=3, command=self.power_law_invoker)
        confirm_button.pack()

    # call back functions for selecting parameters
    def power_law_constant_assign(self, evt):
        w = evt.widget
        self.power_law_constant = w.get()

    def gamma_assign(self, evt):
        w = evt.widget
        self.power_law_gamma = w.get()

    def power_law_invoker(self):
        self.right_show_mask_button.config(command=lambda: mask_invoker(file_path=default_invoker))
        self.right_show_mask_button.pack_forget()
        gamma_arr = power_law(self.GIF_PATH, self.power_law_constant, self.power_law_gamma)
        if gamma_arr is None:
            messagebox.showerror('Invalid Input', 'Please input valid integer or numeric number')
            return
        Image.fromarray(np.uint8(gamma_arr)).save(self.POWER_LAW_PATH)
        self.power_law_image = tk.PhotoImage(file=self.POWER_LAW_PATH)
        self.right_canvas.create_image(20, 20, anchor='nw', image=self.power_law_image)
        self.right_hist_button.config(
            command=lambda: histogram_invoker(file_path=self.POWER_LAW_PATH))
        self.right_hist_button.pack()

    def do_global_histogram_equalization(self):
        self.right_show_mask_button.config(command=lambda: mask_invoker(file_path=default_invoker))
        self.right_show_mask_button.pack_forget()
        equalized_arr = global_histogram_equalization(self.GIF_PATH)
        if equalized_arr is None:
            messagebox.showerror('Unsupported image', 'Please import the right format image')
            return
        Image.fromarray(equalized_arr).save(self.GLOBAL_HISTOGRAM_EQUALIZED_PATH)
        self.global_histogram_equalized_image = tk.PhotoImage(file=self.GLOBAL_HISTOGRAM_EQUALIZED_PATH)
        self.right_canvas.create_image(20, 20, anchor='nw', image=self.global_histogram_equalized_image)
        self.right_hist_button.config(
            command=lambda: histogram_invoker(file_path=self.GLOBAL_HISTOGRAM_EQUALIZED_PATH))
        self.right_hist_button.pack()

    def do_local_histogram_equalization(self):
        self.right_show_mask_button.config(command=lambda: mask_invoker(file_path=default_invoker))
        self.right_show_mask_button.pack_forget()
        equalized_arr = local_histogram_equalization(self.GIF_PATH)
        if equalized_arr is None:
            messagebox.showerror('Unsupported image', 'Please import the right format image')
            return
        Image.fromarray(equalized_arr).save(self.LOCAL_HISTOGRAM_EQUALIZED_PATH)
        self.local_histogram_equalized_image = tk.PhotoImage(file=self.LOCAL_HISTOGRAM_EQUALIZED_PATH)
        self.right_canvas.create_image(20, 20, anchor='nw', image=self.local_histogram_equalized_image)
        self.right_hist_button.config(
            command=lambda: histogram_invoker(file_path=self.GLOBAL_HISTOGRAM_EQUALIZED_PATH))
        self.right_hist_button.pack()

    def do_box_smoothing(self):
        confirm_window = tk.Toplevel(self.root)
        confirm_window.geometry('300x100')
        constant_label = tk.Label(confirm_window, text="Please input box smoothing kernel size:")
        constant_label.pack()
        constant_entry = tk.Entry(confirm_window, show=None)
        constant_entry.bind('<KeyRelease>', self.box_kernel_size_assign)
        constant_entry.pack()
        confirm_button = tk.Button(confirm_window, text='Confirm', width=10, height=3, command=self.box_smoothing_invoker)
        confirm_button.pack()

    # call back function for box kernel size
    def box_kernel_size_assign(self, evt):
        w = evt.widget
        self.box_kernel_size = w.get()

    def box_smoothing_invoker(self):
        self.right_show_mask_button.config(command=lambda: mask_invoker(file_path=default_invoker))
        self.right_show_mask_button.pack_forget()
        box_smoothing_arr = box_smoothing(self.GIF_PATH, self.box_kernel_size)
        if box_smoothing_arr is None:
            messagebox.showerror('Invalid Input', 'Please input valid odd integer size')
            return
        Image.fromarray(np.uint8(box_smoothing_arr)).save(self.BOX_SMOOTHING_PATH)
        self.box_smoothing_image = tk.PhotoImage(file=self.BOX_SMOOTHING_PATH)
        self.right_canvas.create_image(20, 20, anchor='nw', image=self.box_smoothing_image)
        self.right_hist_button.config(
            command=lambda: histogram_invoker(file_path=self.BOX_SMOOTHING_PATH))
        self.right_hist_button.pack()

    def do_gaussian_smoothing(self):
        confirm_window = tk.Toplevel(self.root)
        confirm_window.geometry('300x200')
        kernel_label = tk.Label(confirm_window, text="Please input gaussian smoothing kernel size:")
        kernel_label.pack()
        kernel_entry = tk.Entry(confirm_window, show=None)
        kernel_entry.bind('<KeyRelease>', self.gaussian_kernel_size_assign)
        kernel_entry.pack()
        constant_label = tk.Label(confirm_window, text="Please input gaussian smoothing constant value:")
        constant_label.pack()
        constant_entry = tk.Entry(confirm_window, show=None)
        constant_entry.bind('<KeyRelease>', self.gaussian_constant_assign)
        constant_entry.pack()
        sigma_label = tk.Label(confirm_window, text="Please input gaussian smoothing sigma value:")
        sigma_label.pack()
        sigma_entry = tk.Entry(confirm_window, show=None)
        sigma_entry.bind('<KeyRelease>', self.gaussian_sigma_assign)
        sigma_entry.pack()
        confirm_button = tk.Button(confirm_window, text='Confirm', width=10, height=3, command=self.gaussian_smoothing_invoker)
        confirm_button.pack()

    def gaussian_kernel_size_assign(self, evt):
        w = evt.widget
        self.gaussian_kernel_size = w.get()

    def gaussian_constant_assign(self, evt):
        w = evt.widget
        self.gaussian_constant_k = w.get()

    def gaussian_sigma_assign(self, evt):
        w = evt.widget
        self.gaussian_sigma = w.get()

    def gaussian_smoothing_invoker(self):
        gaussian_smoothing_arr = gaussian_smoothing(self.GIF_PATH, self.gaussian_constant_k, self.gaussian_kernel_size, self.gaussian_sigma)
        if gaussian_smoothing_arr is None:
            messagebox.showerror('Invalid Input', 'Please input valid value')
            return
        Image.fromarray(np.uint8(gaussian_smoothing_arr)).save(self.GAUSSIAN_SMOOTHING_PATH)
        self.gaussian_smoothing_image = tk.PhotoImage(file=self.GAUSSIAN_SMOOTHING_PATH)
        self.right_canvas.create_image(20, 20, anchor='nw', image=self.gaussian_smoothing_image)
        self.right_hist_button.config(
            command=lambda: histogram_invoker(file_path=self.GAUSSIAN_SMOOTHING_PATH))
        self.right_hist_button.pack()

    def do_laplacian_sharping(self):
        laplacian_sharping_arr, laplacian_sharping_mask = laplacian_sharping(self.GIF_PATH)
        if laplacian_sharping_arr is None:
            messagebox.showerror('Unsupported image', 'Please import the right format image')
            return
        Image.fromarray(laplacian_sharping_arr).save(self.LAPLACIAN_SHARPING_PATH)
        Image.fromarray(laplacian_sharping_mask).save(self.LAPLACIAN_SHARPING_MASK_PATH)
        self.laplacian_sharping_image = tk.PhotoImage(file=self.LAPLACIAN_SHARPING_PATH)
        self.right_canvas.create_image(20, 20, anchor='nw', image=self.laplacian_sharping_image)
        self.right_hist_button.config(
            command=lambda: histogram_invoker(file_path=self.LAPLACIAN_SHARPING_PATH))

        self.right_show_mask_button.config(command=lambda: mask_invoker(file_path=self.LAPLACIAN_SHARPING_MASK_PATH))
        self.right_hist_button.pack()
        self.right_show_mask_button.pack()
    
    def do_median_filtering(self):
        self.right_show_mask_button.config(command=lambda: mask_invoker(file_path=default_invoker))
        self.right_show_mask_button.pack_forget()
        median_filtering_arr = order_statistic(self.GIF_PATH, 3)
        if median_filtering_arr is None:
            messagebox.showerror('Invalid Input', 'Please input valid value')
            return
        Image.fromarray(np.uint8(median_filtering_arr)).save(self.MEDIAN_FILTERING_PATH)
        self.median_filtering_image = tk.PhotoImage(file=self.MEDIAN_FILTERING_PATH)
        self.right_canvas.create_image(20, 20, anchor='nw', image=self.median_filtering_image)
        self.right_hist_button.config(
            command=lambda: histogram_invoker(file_path=self.MEDIAN_FILTERING_PATH))
        self.right_hist_button.pack()
    
    def do_highboost(self):
        confirm_window = tk.Toplevel(self.root)
        confirm_window.geometry('300x200')
        kernel_label = tk.Label(confirm_window, text="Please input highboost kernel size:")
        kernel_label.pack()
        kernel_entry = tk.Entry(confirm_window, show=None)
        kernel_entry.bind('<KeyRelease>', self.highboost_kernel_size_assign)
        kernel_entry.pack()
        constant_label = tk.Label(confirm_window, text="Please input highboost constant k value:")
        constant_label.pack()
        constant_entry = tk.Entry(confirm_window, show=None)
        constant_entry.bind('<KeyRelease>', self.highboost_constant_assign)
        constant_entry.pack()
        sigma_label = tk.Label(confirm_window, text="Please input highboost sigma value:")
        sigma_label.pack()
        sigma_entry = tk.Entry(confirm_window, show=None)
        sigma_entry.bind('<KeyRelease>', self.highboost_sigma_assign)
        sigma_entry.pack()
        confirm_button = tk.Button(confirm_window, text='Confirm', width=10, height=3, command=self.highboost__invoker)
        confirm_button.pack()

    def highboost_kernel_size_assign(self, evt):
        w = evt.widget
        self.highboost_kernel_size = w.get()

    def highboost_constant_assign(self, evt):
        w = evt.widget
        self.highboost_constant_k = w.get()

    def highboost_sigma_assign(self, evt):
        w = evt.widget
        self.highboost_sigma = w.get()

    def highboost__invoker(self):
        highboost_arr, highboost_mask = highboost(self.GIF_PATH, self.highboost_constant_k, self.highboost_kernel_size, self.highboost_sigma)
        if highboost_arr is None:
            messagebox.showerror('Invalid Input', 'Please input valid value')
            return
        Image.fromarray(highboost_arr).save(self.HIGHBOOST_PATH)
        Image.fromarray(highboost_mask).save(self.HIGHBOOST_MASK_PATH)
        self.highboost_image = tk.PhotoImage(file=self.HIGHBOOST_PATH)
        self.right_canvas.create_image(20, 20, anchor='nw', image=self.highboost_image)
        self.right_hist_button.config(
            command=lambda: histogram_invoker(file_path=self.HIGHBOOST_PATH))
        self.right_show_mask_button.config(command=lambda: mask_invoker(file_path=self.HIGHBOOST_MASK_PATH))
        self.right_hist_button.pack()
        self.right_show_mask_button.pack()

