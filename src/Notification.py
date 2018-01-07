from abc import ABC, abstractmethod
import sys


class Notifier(ABC):
    activeNotifier = None

    def __init__(self):
        Notifier.activeNotifier = self

    @abstractmethod
    def play_soundfile_async(self, filename):
        print("Cannot play sound file '{}'".format(filename))

    @abstractmethod
    def speak_message(self, message):
        import os
        os.system("Cannot say '{}'".format(message))

    @abstractmethod
    def play_soundfile(self, filename):
        print("Cannot play sound file '{}'".format(filename))


"""
A collection of Macintosh specific notification functions.  
"""


class MacNotifier(Notifier):
    def __init__(self):
        super().__init__()

    def play_soundfile_async(self, filename):
        """
        Play this and don't hang around waiting for it to finish
        Each call spawns an independent playback thread
        :param filename: Path for file to play
        :type filename: basestring
        """
        import threading
        threading._start_new_thread(MacNotifier.play_soundfile, (self, filename))

    def speak_message(self, message):
        """
        Speak message in current system voice
        :param message: Message
        :type message: basestring
        """
        import os
        os.system("say '{}' &".format(message))

    @staticmethod
    def play_soundfile(self, filename):
        """
        Low level soundfile player
        :param filename: Path for file to play
        :type filename: basestring
        """
        CHUNK = 1024
        import wave
        wf = wave.open("../resources/" + filename, 'rb')

        import pyaudio
        p = pyaudio.PyAudio()

        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        data = wf.readframes(CHUNK)

        while len(data) != 0:
            stream.write(data)
            data = wf.readframes(CHUNK)

        stream.stop_stream()
        stream.close()

        p.terminate()
