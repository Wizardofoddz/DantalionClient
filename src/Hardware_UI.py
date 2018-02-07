"""
Low level GUI for talking to camera hardware and low level processes
"""
import json
import threading
import tkinter as tk
from datetime import datetime
import io

import cv2
import os
from PIL import ImageTk, Image, ExifTags

import tkinter as tk

import Hardware_UI as UI
import HardwareAbstractionLayer as HAL
from Notification import MacNotifier
from ProcessAbstractionLayer import IDEFProcess
from ProcessLibrary import IntrinsicsCalculatorProcess, ImageDifferentialCalculatorProcess, StereoCalculatorProcess


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
        size = (1900, 580)
        x = w / 2 - size[0] / 2
        y = h / 2 - size[1] / 2
        MasterControl.root.geometry("%dx%d+%d+%d" % (size + (x, y)))

        MasterControl(MasterControl.root).pack(side="top", fill="both", expand=True)

        MasterControl.root.lift()
        MasterControl.root.attributes('-topmost', True)
        MasterControl.root.after_idle(MasterControl.root.attributes, '-topmost', False)

        MasterControl.root.protocol("WM_DELETE_WINDOW", MasterControl._delete_window)
        MasterControl.root.bind("<Destroy>", MasterControl._destroy)
        while True:
            try:
                MasterControl.root.mainloop()
                break
            except UnicodeDecodeError:
                pass

    @staticmethod
    def _delete_window():
        print("delete_window")
        try:
            IDEFProcess.Run = False
            MasterControl.root.destroy()
        except:
            pass

    @staticmethod
    def _destroy(event):
        print("destroy")

    def __init__(self, parent, *args, **kwargs):
        super(MasterControl, self).__init__(parent)
        MasterControl.application_singleton = self
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.pack(fill=tk.BOTH, expand=tk.YES, side=tk.LEFT)
        self.parent = parent
        # start the notification service - note this is platform specific!
        MacNotifier()

        self.left_imager = None
        self.right_imager = None
        self.common_data = None

        self.programbuttons = None
        self.trigger = False;
        self.stereo_alpha_var = tk.StringVar()
        self.stereo_scale_var = tk.StringVar()
        self.stereo_alpha_var.set('-1')
        self.stereo_alpha_var.trace('w', self.update_stereo_alpha)
        self.stereo_scale_var.set('1')
        self.stereo_scale_var.trace('w', self.update_stereo_scale)
        self.jitter_var = tk.IntVar()
        self.jitter_var.set(0)
        image = ImageTk.PhotoImage(Image.open("../resources/console.jpg"))
        ctl = tk.Label(self, image=image)
        ctl.image = image
        ctl.place(x=0, y=0, relwidth=1, relheight=1)

        self.after(2000, self.open_display)

        HAL.Controller.initialize_controllers(self)
        IDEFProcess.initialize_image_intake(['cancer'])
        IDEFProcess.bind_imager_input_functions(HAL.Controller.Controllers[0].get_left_imager())
        IDEFProcess.bind_imager_input_functions(HAL.Controller.Controllers[0].get_right_imager())
        self.process_thread = threading.Thread(target=IDEFProcess.process_dispatcher)
        self.process_thread.daemon = True
        self.process_thread.start()

    def open_display(self):
        for widget in self.winfo_children():
            widget.destroy()
        # todo Controller selection currently hardwired to zero
        imager_pair = tk.Frame(self)
        self.left_imager = UI.ImagerPanel(imager_pair, 0)
        self.left_imager.pack(fill=tk.BOTH, expand=tk.YES, side=tk.LEFT)
        self.common_data = tk.Frame(imager_pair, width=15)
        tk.Label(self.common_data, text="Jitter").pack(side=tk.TOP)
        tk.Label(self.common_data, textvariable=self.jitter_var).pack(side=tk.TOP)
        self.programbuttons = tk.Frame(self.common_data)
        tk.Button(self.programbuttons, width=10, text="SterCalib", command=self.start_stereo_calibration).pack(
            side=tk.TOP)
        tk.Button(self.programbuttons, width=10, text="Set Stereo", command=self.set_stereo_calibration).pack(
            side=tk.TOP)
        tk.Button(self.programbuttons, width=10, text="Trigger", command=self.set_trigger).pack(side=tk.TOP)
        tk.Button(self.programbuttons, width=10, text="Purge Sessions", command=self.purge_sessions).pack(side=tk.TOP)
        tk.Button(self.programbuttons, width=10, text="Export CSV", command=self.create_calibration_csv).pack(side=tk.TOP)
        self.programbuttons.pack()
        tk.Label(self.common_data, text="Alpha").pack(side=tk.TOP)
        tk.Entry(self.common_data, textvariable=self.stereo_alpha_var).pack(side=tk.TOP)
        tk.Label(self.common_data, text="Scale").pack(side=tk.TOP)
        tk.Entry(self.common_data, textvariable=self.stereo_scale_var).pack(side=tk.TOP)
        self.common_data.pack(fill=tk.Y, expand=tk.NO, side=tk.LEFT)
        self.right_imager = UI.ImagerPanel(imager_pair, 1)
        self.right_imager.pack(fill=tk.BOTH, expand=tk.YES, side=tk.RIGHT)
        imager_pair.pack(fill=tk.BOTH, expand=tk.YES, side=tk.TOP)
        bottomrow = tk.Frame(self)
        self.status_line = tk.Label(bottomrow, text="Main Message Display")
        self.status_line.pack(side=tk.LEFT)
        bottomrow.pack(side=tk.TOP)
        self.update_status_display()

    def set_trigger(self):
        self.trigger = True

    def purge_sessions(self):
        sidir = StereoCalculatorProcess.stereo_session_image_folder()
        list(map(os.unlink, (os.path.join(sidir, f) for f in os.listdir(sidir))))
        ssdir = StereoCalculatorProcess.stereo_sessionfolder()
        list(map(os.unlink, (os.path.join(ssdir, f) for f in os.listdir(ssdir))))

    def update_stereo_alpha(self, a, b, c):
        alpha = float(self.stereo_alpha_var.get())
        HAL.Controller.Controllers[0].publish_message(HAL.Controller.Controllers[0].imagers[0].imager_address,
                                                      'stereo_rectify_alpha', alpha)
    def update_stereo_scale(self, a, b, c):
        scale = int(self.stereo_scale_var.get())
        HAL.Controller.Controllers[0].publish_message(HAL.Controller.Controllers[0].imagers[0].imager_address,
                                                      'stereo_rectify_scale', scale)
    def stereo_calibration_filename(self):
        """
        Location of precomputed calibration file with camera intrinsics and stereo extrinsics
        :return: Location of precomputed calibration file with camera intrinsics
        :rtype: basestring
        """
        filename = "stereo_{}_0.json".format(HAL.Controller.Controllers[0].resource)
        return "../resources/" + filename

    def set_stereo_calibration(self):
        filename = self.stereo_calibration_filename()
        with open(filename, "r") as file:
            cfg = file.read()
            HAL.Controller.Controllers[0].publish_message(0, "stereoextrinsics", cfg)

    def start_stereo_calibration(self):
        leftimager = HAL.Controller.Controllers[0].imagers[0]
        rightimager = HAL.Controller.Controllers[0].imagers[1]

        icp = StereoCalculatorProcess("SterCalib", leftimager, rightimager)
        icp.initialize({"RECORDING" : False})
        icp.status_message = self.change_status_message
        IDEFProcess.add_process(icp)

    def update_status_display(self):
        self.jitter_var.set(IDEFProcess.Jitter)
        for name, button in self.programbuttons.children.items():
            label = button['text']
            if label in IDEFProcess.ActiveProcesses:
                button.configure(highlightbackground='blue')
            else:
                button.configure(highlightbackground='green')
        self.update()
        self.after(2000, self.update_status_display)

    def change_status_message(self, text):
        """
        Replace status line text with the provided text
        :param text:
        :type text:
        :return:
        :rtype:
        """
        self.status_line.config(text=text)

    def create_calibration_csv(self):
        with open('../CalibrationRecords/intrinsicsHistory.txt', 'r') as myfile:
            data = '[' + myfile.read() + ']'
        jd = json.loads(data)
        with open('../CalibrationRecords/intrinsicsHistory.csv', 'w') as myfile:
            for x in jd:
                p1 = '{},{},{},{},{},{},{}'.format(x['ID'],x['TIMESTAMP'],x['CONTROLLER'],x['CAMERAINDEX'],
                                       x['MATCHCOUNT'],x['MATCHSEPARATION'],x['REPROJECTIONERROR'])
                cmstr = ''
                for i in range(0,9):
                    if i != 0:
                        cmstr += ','
                    cmstr += str(x['CAMERAMATRIX'][2 + i])

                dsstr = ''
                for i in range(0,5):
                    if i != 0:
                        dsstr += ','
                    dsstr += str(x['DISTORTIONCOEFFICIENTS'][2 + i])
                line = p1 + ',' + cmstr + ',' + dsstr + '\n'
                myfile.write(line)

class ImagerPanel(tk.Frame):
    @staticmethod
    def image_click(event):
        # event.widget.get_imager().toggle_camera()
        print('livk')

    def get_imager(self):
        return HAL.Controller.Controllers[0].imagers[self.image_index.get()]

    def __init__(self, parent, imagerIndex, *args, **kwargs):
        tk.Frame.__init__(self, parent, padx=10, pady=10, bd=5, relief=tk.RAISED)
        self.image_index = tk.IntVar()
        self.image_index.set(imagerIndex)
        self.serial_update_index = 0

        "top row that carries info"
        toprow = tk.Frame(self)
        tk.Label(toprow,text = "Ctlr: {} ".format(self.get_imager().controller.resource)).pack(side=tk.LEFT)
        tk.Radiobutton(toprow,
                       text="LEFT",
                       width=7,
                       padx=0,
                       variable=self.image_index,
                       value=0).pack(side=tk.LEFT)
        tk.Radiobutton(toprow,
                       text="RIGHT",
                       width=10,
                       padx=3,
                       variable=self.image_index,
                       value=1).pack(side=tk.LEFT)

        self.label_channel = tk.Label(toprow, text="UNKNOWN")
        self.label_channel.pack(side=tk.LEFT)

        choices = ['raw']
        self.selected_imager_channel = tk.StringVar(toprow)
        self.selected_imager_channel.set('raw')
        self.num_calibration_samples = tk.StringVar()
        self.num_calibration_samples.set("24")
        self.imager_channels = tk.OptionMenu(toprow, self.selected_imager_channel, *choices)
        self.imager_channels.pack(side=tk.LEFT)

        self.label_framerate = tk.Label(toprow, text="0000")
        self.label_framerate.pack(side=tk.RIGHT)

        toprow.pack(side=tk.TOP)

        centerrow = tk.Frame(self)
        leftcol = tk.Frame(centerrow)
        self.programbuttons = tk.Frame(leftcol)
        tk.Button(self.programbuttons, width=10, text="Calibrate", command=self.start_calibration).pack(side=tk.TOP)
        tk.Button(self.programbuttons, width=10, text="Uncalibr", command=self.stop_calibration).pack(side=tk.TOP)
        tk.Button(self.programbuttons, width=10, text="Diff", command=self.start_differential).pack(side=tk.TOP)
        tk.Button(self.programbuttons, width=10, text="Reset FPS", command=self.get_imager().reset_fps_samples).pack(
            side=tk.TOP)
        self.programbuttons.pack(side=tk.TOP)
        df = tk.Frame(leftcol)
        tk.Label(df, text="#SAM").pack(side=tk.LEFT)
        tk.Entry(df,width=5, textvariable=self.num_calibration_samples).pack(side=tk.LEFT)
        df.pack(side=tk.TOP)
        leftcol.pack(side=tk.LEFT)

        canvas_frame = tk.Frame(centerrow)
        self.imager_display = tk.Canvas(canvas_frame, bd=6, highlightthickness=3, scrollregion=(0, 0, 500, 500))
        hbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        hbar.config(command=self.imager_display.xview)
        vbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        vbar.config(command=self.imager_display.yview)
        self.imager_display.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.imager_display.image_id = None
        self.imager_display.imager = self.get_imager()
        self.imager_display.imager.updateNotifier = self.update_image

        self.imager_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)
        # camera control buttons
        self.controlbuttons = tk.Frame(centerrow)
        self.build_control_buttons()
        self.controlbuttons.pack(side=tk.LEFT)

        centerrow.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        bottomrow = tk.Frame(self)
        self.status_line = tk.Label(bottomrow, text="Imager Message Display")
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
                  command=self.get_imager().reboot).pack(side=tk.TOP)
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
        self.get_imager().controller.publish_message(self.get_imager().imager_address, control_name, set_value)

    def update_image(self):
        """
        Perform update of raw image - expected to be asyncronously driven
        :return:
        :rtype:
        """
        cv_image = self.get_imager().get_image(self.selected_imager_channel.get(), False)
        if cv_image is None:
            return
        now = datetime.now()
        if self.get_imager().last_time is not None:
            self.get_imager().framerate_sum += (now - self.get_imager().last_time)
        self.get_imager().last_time = now

    def start_calibration(self):
        # we want undistorted
        # if 'UNDISTORT' not in self.get_imager().controller.imagers[self.get_imager().imager_address].processes:
        icp = IntrinsicsCalculatorProcess('IntrinsicsCalculatorProcess', self.get_imager())
        icp.initialize({'MATCHCOUNT': int(self.num_calibration_samples.get())})
        icp.status_message = self.change_status_message
        IDEFProcess.ActiveProcesses['Calibrate'] = icp

    def stop_calibration(self):
        self.get_imager().controller.publish_message(self.get_imager().imager_address, "intrinsics", None)

    def get_calibration(self):
        return self.get_imager().get_calibration()

    def set_calibration(self):
        self.get_imager().set_calibration()

    def start_differential(self):
        # we want undistorted
        # if 'UNDISTORT' not in self.get_imager().controller.imagers[self.get_imager().imager_address].processes:
        icp = ImageDifferentialCalculatorProcess('ImageDifferentialCalculatorProcess', self.get_imager())
        icp.initialize({})
        icp.status_message = self.change_status_message
        IDEFProcess.ActiveProcesses['Differential'] = icp

    # def get_imager(self, imager_ordinal):
    #    return self.imager_displays[imager_ordinal]

    def status_update(self):
        """
        Update the on screen status information
        :return:
        :rtype:
        """
        self.update_imager_status_display(0)
        self.after(1000, self.status_update)

    def capture_update(self):
        """
        Update the latest image
        :return:
        :rtype:
        """
        """
        Update the current image for the attached imager display
        :return:
        :rtype:
        """
        try:
            rawimage = self.get_imager().get_image(self.selected_imager_channel.get(), False)
            if rawimage is None:
                return

            self.imager_display.image = ImageTk.PhotoImage(Image.open(io.BytesIO(rawimage)))
            if self.imager_display.image_id is None:
                self.imager_display.image_id = self.imager_display.create_image(0, 0, image=self.imager_display.image,
                                                                                anchor='nw')
            else:
                self.imager_display.itemconfigure(self.imager_display.image_id, image=self.imager_display.image)

            self.imager_display.configure(scrollregion=self.imager_display.bbox("all"))
        except:
            pass
        finally:
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

    def update_imager_status_display(self, imager_ordinal):
        """ IFF there is an available image, show it in the imager display

        :param imager_ordinal: identifies the controller
        :type imager_ordinal: int
        """

        if self.get_imager().framerate_sum.total_seconds() == 0:
            return
        persecond = self.get_imager().framerate_sum.total_seconds() / self.get_imager().local_image_counter

        self.label_framerate.config(text="{0:.2f}".format(1.0 / persecond))

        # self.label_channel.config(text=self.get_imager().channel)

        self.imager_channels['menu'].delete(0, 'end')
        for choice in self.get_imager().raw_image.keys():
            self.imager_channels['menu'].add_command(label=choice,
                                                     command=tk._setit(self.selected_imager_channel, choice))

        for name, button in self.programbuttons.children.items():
            label = button['text']
            if label in IDEFProcess.ActiveProcesses:
                button.configure(highlightbackground='blue')
            else:
                button.configure(highlightbackground='green')
