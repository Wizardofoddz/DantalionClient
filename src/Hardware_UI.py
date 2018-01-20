"""
Low level GUI for talking to camera hardware and low level processes
"""
import tkinter as tk
from datetime import datetime
import io

import cv2
from PIL import ImageTk, Image, ExifTags

import tkinter as tk

import Hardware_UI as UI
import HardwareAbstractionLayer as HAL
from Notification import MacNotifier
from ProcessLibrary import IntrinsicsCalculatorProcess, ImageDifferentialCalculatorProcess


class MasterControl(tk.Frame):
    """
    The top level application window for the application
    """
    root = None
    application_singleton = None
    @staticmethod
    def run():

        # if __name__ == "__main__":
        MasterControl.root = tk.Tk()
        MasterControl.root.title("Camera Calibration")
        MasterControl.root.resizable(width=True, height=True)

        MasterControl.root.geometry('{}x{}'.format(1, 1))
        MasterControl.root.update_idletasks()
        w = MasterControl.root.winfo_screenwidth()
        h = MasterControl.root.winfo_screenheight()
        size = (1024, 580)
        x = w / 2 - size[0] / 2
        y = h / 2 - size[1] / 2
        MasterControl.root.geometry("%dx%d+%d+%d" % (size + (x, y)))

        MasterControl(MasterControl.root).pack(side="top", fill="both", expand=True)

        MasterControl.root.lift()
        MasterControl.root.attributes('-topmost', True)
        MasterControl.root.after_idle(MasterControl.root.attributes, '-topmost', False)

        while True:
            try:
                MasterControl.root.mainloop()
                break
            except UnicodeDecodeError:
                pass

    def __init__(self, parent, *args, **kwargs):
        super(MasterControl, self).__init__(parent)
        application_singleton = self
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.pack(fill=tk.BOTH, expand=tk.YES, side=tk.LEFT)
        self.parent = parent
        # start the notification service - note this is platform specific!
        MacNotifier()

        self.alpha_imager = None
        self.beta_imager = None
        self.common_data = None

        self.jitter_var = tk.IntVar()
        self.jitter_var.set(0)
        image = ImageTk.PhotoImage(Image.open("../resources/console.jpg"))
        ctl = tk.Label(self, image=image)
        ctl.image = image
        ctl.place(x=0, y=0, relwidth=1, relheight=1)

        self.after(2000, self.open_display)

        HAL.Controller.initialize_controllers(self)

    def open_display(self):
        for widget in self.winfo_children():
            widget.destroy()
        # todo Controller selection currently hardwired to zero
        self.alpha_imager = UI.ImagerPanel(self, HAL.Controller.Controllers[0].imagers[0])
        self.alpha_imager.pack(fill=tk.BOTH, expand=tk.YES, side=tk.LEFT)
        self.common_data = tk.Frame(self, width=15)
        tk.Label(self.common_data, text="Jitter").pack(side=tk.TOP)
        tk.Label(self.common_data, textvariable=self.jitter_var).pack(side=tk.TOP)
        self.common_data.pack(fill=tk.Y, expand=tk.NO, side=tk.LEFT)
        self.beta_imager = UI.ImagerPanel(self, HAL.Controller.Controllers[0].imagers[1])
        self.beta_imager.pack(fill=tk.BOTH, expand=tk.YES, side=tk.RIGHT)


class ImagerPanel(tk.Frame):
    @staticmethod
    def image_click(event):
        # event.widget.imager.toggle_camera()
        print('livk')

    def __init__(self, parent, imager, *args, **kwargs):
        tk.Frame.__init__(self, parent, padx=10, pady=10, bd=5, relief=tk.RAISED)
        self.imager = imager
        self.serial_update_index = 0

        "top row that carries info"
        toprow = tk.Frame(self)
        tk.Label(toprow, text="{}:{}".format(self.imager.controller.resource,
                                             'A' if self.imager.imager_address == 0 else 'O')).pack(
            side=tk.LEFT)
        self.label_channel = tk.Label(toprow, text="UNKNOWN")
        self.label_channel.pack(side=tk.LEFT)

        choices = ['raw']
        self.selected_imager_channel = tk.StringVar(toprow)
        self.selected_imager_channel.set('raw')
        self.imager_channels = tk.OptionMenu(toprow, self.selected_imager_channel, *choices)
        self.imager_channels.pack(side=tk.LEFT)

        self.label_framerate = tk.Label(toprow, text="0000")
        self.label_framerate.pack(side=tk.RIGHT)

        toprow.pack(side=tk.TOP)

        centerrow = tk.Frame(self)
        self.programbuttons = tk.Frame(centerrow)
        tk.Button(self.programbuttons, width=10, text="Calibrate", command=self.start_calibration).pack(side=tk.TOP)
        tk.Button(self.programbuttons, width=10, text="Uncalibr", command=self.stop_calibration).pack(side=tk.TOP)
        tk.Button(self.programbuttons, width=10, text="Diff", command=self.start_differential).pack(side=tk.TOP)
        tk.Button(self.programbuttons, width=10, text="Reset FPS", command=self.imager.reset_fps_samples).pack(
            side=tk.TOP)
        self.programbuttons.pack(side=tk.LEFT)
        self.imager_display = tk.Canvas(centerrow, bd=6, highlightthickness=3)

        self.imager_display.image_id = None
        self.imager_display.imager = self.imager
        self.imager_display.imager.display = self.imager_display
        self.imager_display.imager.updateNotifier = self.update_image

        self.imager_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

        # camera control buttons
        self.controlbuttons = tk.Frame(centerrow)
        self.build_control_buttons()
        self.controlbuttons.pack(side=tk.LEFT)

        centerrow.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        bottomrow = tk.Frame(self)
        self.status_line = tk.Label(bottomrow, text="Hi there!")
        self.status_line.pack(side=tk.LEFT)
        bottomrow.pack(side=tk.TOP)
        self.status_update()
        self.capture_update()

    def build_control_buttons(self):
        """
        Create and attach all of the camera control buttons
        :return: self changed as side effect
        :rtype:
        """
        tk.Button(self.controlbuttons, width=10, text="REBOOT",
                  command=self.imager.reboot).pack(side=tk.TOP)
        tk.Button(self.controlbuttons, width=10, text="Set Intrins",
                  command=self.set_calibration).pack(side=tk.TOP)
        tk.Scale(self.controlbuttons, width=10, from_=4, to=63, orient=tk.HORIZONTAL, tickinterval=16,
                 command=self.set_quantization).pack(side=tk.TOP)
        rez = [('160x120', 1), ('176x144', 2), ('320x240', 3), ('352x288', 4),
               ('640x480', 5), ('800x600', 6), ('1024x768', 7), ('1280x1024', 8),
               ('1600x1200', 9), ('1280x960', 10), ('2048x1536', 11), ('2592x1944', 12)]
        self.build_control_menu(self.controlbuttons, "Resolution", "resolution", rez)
        lm = [('auto', 0), ('sun', 1), ('cloud', 2), ('office', 3), ('home', 4)]
        self.build_control_menu(self.controlbuttons, "Light", "lightmode", lm)
        self.build_fixed_level_menu(self.controlbuttons, "Bright", "brightness")
        self.build_fixed_level_menu(self.controlbuttons, "Contrast", "contrast")
        self.build_fixed_level_menu(self.controlbuttons, "Saturate", "colorsaturation")
        sem = [('Antique', 0), ('Bluish', 1), ('Greenish', 2), ('Reddish', 3), ('B/W', 4), ('Negative', 5),
               ('BWNegative', 6), ('Normal', 7), ('Sepia', 8), ('OverExpose', 9), ('Solatize', 10), ('WHAT?', 11),
               ('Yellowish', 12)]
        self.build_control_menu(self.controlbuttons, "Special", "specialeffects", sem)
        state = [('Normal', 0), ('Flip', 1), ('Mirror', 2), ('Both', 3)]
        self.build_control_menu(self.controlbuttons, "Orient", "mirrorflip", state)
        yuv = [('Y U Y V', 0), ('Y V Y U', 1), ('V Y U Y', 2), ('U Y V Y', 3)]
        self.build_control_menu(self.controlbuttons, "YUV", "setyuv", yuv)

    def set_quantization(self, quantizeLevel):
        self.set_camera_control('quantization', quantizeLevel)

    def build_fixed_level_menu(self, host_menu, function_name, control_name):
        ml = [('+2', 0), ('+1', 1), ('0', 2), ('-1', 3), ('-2', 4)]
        return self.build_control_menu(host_menu, function_name, control_name, ml)

    def build_control_menu(self, host_menu, function_name, control_name, menu_items):
        ebutton = tk.Menubutton(host_menu, text=function_name, underline=0)
        ebutton.pack(side=tk.TOP)
        edit = tk.Menu(ebutton, tearoff=0)
        for mtext, cval in menu_items:
            edit.add_command(label=mtext, command=lambda cval=cval: self.set_camera_control(control_name, cval),
                             underline=0)
        ebutton.config(menu=edit)
        return ebutton

    def set_camera_control(self, control_name, set_value):
        print('set {} to {}'.format(control_name, set_value))
        self.imager.controller.publish_message(self.imager.imager_address, control_name, set_value)

    def update_image(self):
        """
        Perform update of raw image - expected to be asyncronously driven
        :return:
        :rtype:
        """
        cv_image = self.imager.get_image(self.selected_imager_channel.get(), False)
        if cv_image is None:
            return

        now = datetime.now()
        if self.imager.last_time is not None:
            self.imager.framerate_sum += (now - self.imager.last_time)
        self.imager.last_time = now

    def start_calibration(self):
        # we want undistorted
        # if 'UNDISTORT' not in self.imager.controller.imagers[self.imager.imager_address].processes:
        icp = IntrinsicsCalculatorProcess('IntrinsicsCalculatorProcess', self.imager)
        icp.initialize()
        icp.status_message = self.change_status_message
        self.imager.processes['Calibrate'] = icp

    def stop_calibration(self):
        self.imager.controller.publish_message(self.imager.imager_address, "intrinsics", None)

    def set_calibration(self):
        filename = self.imager.calibration_filename()
        with open(filename, "r") as file:
            cfg = file.read()
            self.imager.controller.publish_message(self.imager.imager_address, "intrinsics", cfg)

    def start_differential(self):
        # we want undistorted
        # if 'UNDISTORT' not in self.imager.controller.imagers[self.imager.imager_address].processes:
        icp = ImageDifferentialCalculatorProcess('ImageDifferentialCalculatorProcess', self.imager)
        icp.initialize()
        icp.status_message = self.change_status_message
        self.imager.processes['Differential'] = icp

    # def get_imager(self, imager_ordinal):
    #    return self.imager_displays[imager_ordinal]

    def status_update(self):
        """
        Update the on screen status information
        :return:
        :rtype:
        """
        self.update_imager_status_display(0)
        self.update()
        self.after(1000, self.status_update)

    def capture_update(self):
        """
        Update the latest image
        :return:
        :rtype:
        """
        self.update_imager_capture_display()
        self.after(100, self.capture_update)

    def change_status_message(self, text):
        """
        Replace status line text with the provided text
        :param text:
        :type text:
        :return:
        :rtype:
        """
        self.status_line.config(text=text)

    def update_imager_capture_display(self):
        """
        Update the current image for the attached imager display
        :return:
        :rtype:
        """
        try:
            channelname = self.selected_imager_channel.get()
            rawimage = self.imager.get_image(channelname, False)
            if rawimage is None:
                return
            # if channelname == "rectified":
            #     new_file = open("camimg.jpg", "wb")
            #     new_file.write(rawimage)
            #     new_file.close()

            # image = Image.fromarray(cv2.cvtColor(rawimage, cv2.COLOR_BGR2RGB))
            image = Image.open(io.BytesIO(rawimage))

            temp = ImageTk.PhotoImage(image)
            self.imager.display.image = temp
            if self.imager.display.image_id is None:
                self.imager.display.image_id = self.imager.display.create_image(0, 0, image=self.imager.display.image,
                                                                                anchor='nw')
            else:
                self.imager.display.itemconfigure(self.imager.display.image_id, image=self.imager.display.image)
        except:
            pass

    def update_imager_status_display(self, imager_ordinal):
        """ IFF there is an available image, show it in the imager display

        :param imager_ordinal: identifies the controller
        :type imager_ordinal: int
        """

        if self.imager.framerate_sum.total_seconds() == 0:
            return
        persecond = self.imager.framerate_sum.total_seconds() / self.imager.local_image_counter

        self.label_framerate.config(text="{0:.2f}".format(1.0 / persecond))

        #self.label_channel.config(text=self.imager.channel)

        self.imager_channels['menu'].delete(0, 'end')
        for choice in self.imager.raw_image.keys():
            self.imager_channels['menu'].add_command(label=choice,
                                                     command=tk._setit(self.selected_imager_channel, choice))

        for name, button in self.programbuttons.children.items():
            label = button['text']
            if label in self.imager.processes:
                button.configure(highlightbackground='blue')
            else:
                button.configure(highlightbackground='green')
