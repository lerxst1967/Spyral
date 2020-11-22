# -*- coding: utf-8 -*-
"""
Spyral Music Visualizer, version 1.06

Last Modified on Oct 21 2020

@author: MJ Hurben  / hurbenm@gmail.com
"""

my_text=("Spyral Music Visualizer\n \
Version 1.06 Oct 21 2020\n \
MJ Hurben hurbenm@gmail.com \n \
 \n \
This visualization tool was developed to play WAV files. It can also \
use mp3 or flac files, HOWEVER: It can only do so by first making a WAV from \
the mp3 or flac file. By selecting, for example, 'my_song.mp3', a file named \
'my_song.wav' will be created in the same folder. \n \
\n \
After clicking OK the screen may resize several times. This program \
was written to run in fullscreen mode. Once playback has started, \
hitting ANY KEY will pause the music and bring up a menu that will \
allow some parameters to be adjusted; or you may restart the program \
if you want to choose a different file or different folder. Once a file \
or folder has been played, this program will exit. Otherwise it can \
be stopped during play by hitting any key and then choosing QUIT. \n \
\n \
The 'Chunk Size' paraemeter is the length, in samples, that is used for \
each FFT. The default value of 12000 seems to work fine, but it can be \
adjusted.... be warned it may cause erratic behavior. By increasing this \
value, there will be more resolution of the lower frequencies. However \
it also increases the time needed to do the FFT. There is always a trade \
off. The spiral scale includes 88 notes corresponding to the 88 piano keys, \
with the lowest note having the smallest radius. By choosing 'Reverse', the \
highest note will have the smallest radius. All notes with the same name lie \
along the same angle, with vertical corresponding to 'A'. Stereo location is \
indicated by color, with red and blue being the extremes. There are three \
options for the central color: green, yellow, or white. \n \
\n \
Attempts to run files other than mp3, flac or wav will likely cause crashes. \
There has been no effort made to make the error handling very sophisticated, \
yet.")    

import pygame, sys
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import wavio
import PySimpleGUI as sg
import os
import subprocess
from pygame.locals import *


def get_stereo_amps(
    chunk_left,
    chunk_right,
    delta_freq,
    
    rel_vol,
    hi_sens,
    hi_sep,
    samp_rate,
    color_map
):

    """  
    The primary function that takes a chunk of time series data, runs the 
    fft, and returns an array of color values for each note on the scale.
    This function is called for succesive "chunks" of wav file data, 
    corresponding to the time marker indicating the current portion of the
    wav file being played
    
    Parameters                                                                                
    ----------                                                                                
    chunk_left : left channel time series chunk data from WAV
    chunk_right : right channel time series chunk data from WAV
    delta_freq : frequency difference between succesive values in the fft array
    rel_vol : relative volume of given chunk compared to maximum overall 
        for the entire wav file
    hi_sens : higher sensitivity flag, allows quieter components to be brighter
        this is done by taking the square root of the spectrum, making the 
        peaks have more similar amplitudes
    hi_sep : higher separation, allows stronger stereo color visualization
    samp_rate : wav file sampling rate
                                                                                                                                                                                                                    
    Returns                                                                                   
    ------- 
    colors_255 : array of color values corresponding to each of the 88 piano 
        key notes
      
    """

    try:
        Pxxl, freq_array, dummy = plt.magnitude_spectrum(
            chunk_left, samp_rate
        )
        plt.close()
        Pxxr, freq_array, dummy = plt.magnitude_spectrum(
            chunk_right, samp_rate
        )
        plt.close()
    except:
        print("exception in getting the frequency spectrum data")

    amps_l = []
    amps_r = []

    for n in range(1, 89):
        ampll = get_amp(n, Pxxl, delta_freq)
        amplr = get_amp(n, Pxxr, delta_freq)

        if hi_sens:
            amps_l.append(np.sqrt(ampll))
            amps_r.append(np.sqrt(amplr))
        else:
            amps_l.append(ampll)
            amps_r.append(amplr)

    colors_255 = get_colors(amps_l, amps_r, rel_vol, hi_sep, color_map)

    return colors_255


def get_amp(note, fft_data, delta_freq):

    """  
    Function that gets the amplitude corresponding of the spectrum for the 
    frequency for a given note. This must be called for each channel
    Formula for "indx" is classic relationship of frequency and piano key #
        
    Parameters                                                                                
    ----------                                                                                
    note : integer from 1 to 88 corresponding to the piano key #
    fft_data : the array of spectrum amplitudes, from fft, for a single channel
    delta_freq : frequency difference between succesive values in the fft array
                                                                                                                                                                                                                          
    Returns                                                                                   
    ------- 
    ampl : amplitude of spectrum corresponding to frequency of given note
      
    """

    indx = int((440 / delta_freq) * (2 ** ((note - 49) / 12)))
    ampl = fft_data[indx]
    return ampl


def get_colors(fft_left, fft_right, rel_vol, hi_sep, color_map):

    """  
    Function that takes left and right fft arrays and computes color values,
    which is based on element-wise difference between the arrays, thereby
    givien stero information, and the alpha value, which determines brightness,
    and which scales the average amplitude for each frequency
    
    Parameters                                                                                
    ----------                                                                                
    fft_left : left amplitude
    fft_right : right amplitude
    rel_vol : max volume of chunk / max volume of wav file
    hi_sep :  if False, uses linear relationship between the stereo difference
        and resulting color. If True, uses sin(x) function instead
                                                                                                                                                                                                                          
    Returns                                                                                   
    ------- 
    color255 : array of color values
      
    """

    max_r = max(fft_right)
    max_l = max(fft_left)
    if color_map==0:
        cmap = cm.rainbow
    elif color_map==1:
        cmap = cm.Spectral
    else:
        cmap = cm.bwr

    color255 = []
    max_lr = max(max_r, max_l)
    lena = len(fft_left)
    for i in range(0, lena):
        try:
            mean_lr = (fft_left[i] - fft_right[i]) / (
                fft_left[i] + fft_right[i]
            )
        except:
            mean_lr = 0
        if hi_sep:
            col_lr = 0.5 * np.sin(3.14 * mean_lr / 2) + 0.5
        else:
            col_lr = 0.5 * mean_lr / 2 + 0.5
        max_amp = max(fft_left[i], fft_right[i])
        if rel_vol > 0.02:
            multi = 1
        else:
            multi = 0
        try:
            alpha = (
                multi * 255 * np.sin(3.14 * max_amp / (2 * max_lr)) ** 2
            )
        except:
            alpha = 127
        color_raw = cmap(col_lr)
        color255.append(
            (
                255 * color_raw[0],
                255 * color_raw[1],
                255 * color_raw[2],
                alpha,
            )
        )

    return color255


def draw_arc(note, color, reverse=False):

    """  
    Function that draws an arc for a specified note and color. Calculation
    is for a simple spiral shape in radial coordiantes, with radius 
    proportional to frequency. The spiral is partitioned into 12 equal sections
    corresponding to musical octaves. Some hard-coded parameters below could
    be tweaked to change size and line thickness.
    
    Parameters                                                                                
    ----------                                                                                
    note : integer from 1 to 88 corresponding to piano key #
    color : RGB and alpha value
    reverse : If True, high frequency will correspond to smaller radius
                                                                                                                                                                                                                          
    Returns                                                                                   
    ------- 
    True
      
    """

    if reverse:
        note = 89 - note

    R0 = INITHEIGHT / 5

    deg_i = 90 - (note - 1) * 30

    while deg_i < 0:
        deg_i = 360 + deg_i
    deg_f = deg_i - 30
    if deg_f < 0:
        deg_f = 360 + deg_f
    theta_i = np.radians(deg_i + 15)
    theta_f = np.radians(deg_f + 15)
    delta_theta = abs(note * 30 * 35)
    r_i = (80 * R0 + delta_theta) ** 0.5

    widthx = int(4400 / r_i)
    if widthx > r_i:
        widthx = int(r_i)
    widthx = int(widthx / 1.0)

    width_box = 2 * r_i
    x = x_0 - r_i
    y = y_0 - r_i

    pygame.draw.arc(
        background,
        color,
        pygame.Rect(x, y, width_box, width_box),
        theta_f,
        theta_i,
        widthx,
    )
    return True

# Main Program

sg.theme("Dark Blue 3")
list_songs = []
file_path = ""
folder_path = ""
TUNE_T = -16384 # Offset used to make visual and auditory timing optimal
NFFT = 12000 # Chunk length, which will be number of points in FFT
INITHEIGHT = 800 # Start window height and width
INITWIDTH = 1500
x_0 = INITWIDTH / 2 # x_0 and y_0 used to place spiral in center of screen
y_0 = INITHEIGHT / 2
menu_num = 0
main_loop = True
color_map=0

while main_loop:
    if menu_num == 0:
        layout = [
            [sg.Text("Select folder or single WAV or MP3 file")],
            [
                sg.Text("Folder", size=(13, 1)),
                sg.InputText(),
                sg.FolderBrowse(),
            ],
            [
                sg.Text("Single File ", size=(13, 1)),
                sg.InputText(),
                sg.FileBrowse(),
            ],
            [
                sg.Checkbox("Reverse Direction", default=False),
                sg.Checkbox("Higher Sensitivity", default=True),
                sg.Checkbox("Greater Separation", default=True),
            ],
            [sg.Text("Chunk Size ", size=(10,1)), 
             sg.Input(NFFT,size=(7,1)), 
             sg.Text("High values boost res of low freqs", size=(36,1)) ],
            [
            sg.Text("Color Scheme :", size=(10,1)),
            sg.Rad('Blue Green Red', 1, default=True), 
            sg.Rad('Red Yellow Blue', 1, default=False),
            sg.Rad('Blue White Red', 1, default=False)
            ], 
            [sg.Button("OK"), sg.Button("QUIT"), sg.Button("README")],
            [
                sg.Text(
                    "Play starts on OK; hit any key to pause and bring up menu"
                )
            ],
        ]

        window = sg.Window("Welcome to Spyral", layout)
        while True:
            event, values = window.read()
            if event=="README":
                sg.popup_scrolled(my_text,title="README")
            elif event == "QUIT" or event == sg.WIN_CLOSED:
                window.close()
                main_loop = False
                do_loop = False
                break
            elif event=="OK":
                break
                
        window.close()
        folder_path, file_path, reverse, hi_sens, hi_sep, NFFT, r1,r2,r3 = (
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            int(values[5]),
            values[6],
            values[7],
            values[8]
        )
        if r1:
            color_map=0
        elif r2:
            color_map=1
        else:
            color_map=2
        if len(folder_path) == 0:
            filename = file_path
            Num_wav = 1
            if filename.split(".")[1] == "mp3" or filename.split(".")[1] == "flac":
                wavfilenamex = filename.split(".")[0] + ".wav"
                if not os.path.isfile(wavfilenamex):
                    subprocess.call(
                        ["ffmpeg", "-i", filename, wavfilenamex]
                    )
                    filename = wavfilenamex
        else:

            Num_wav = 0

            for filex in os.listdir(folder_path):
                print (filex)
                if filex.split(".")[1] == "mp3" or filex.split(".")[1] == "flac":
                    wavfilenamex = (
                        folder_path + "/" + filex.split(".")[0] + ".wav"
                    )
                    wavfilenamex2 = (
                    filex.split(".")[0] + ".wav"
                    )    
                    if not os.path.isfile(wavfilenamex2):
                        fullx = folder_path + "/" + filex
                        subprocess.call(
                            ["ffmpeg", "-i", fullx, wavfilenamex]
                        )

            for filex in os.listdir(folder_path):
                if filex.split(".")[1] == "wav":
                    Num_wav += 1
                    full_song = folder_path + "/" + filex
                    list_songs.append(full_song)
        menu_num = 1
    do_loop = True
    restrt = False
    while do_loop:
        for jjj in range(0, Num_wav):
            pygame.init()

            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

            width, height = screen.get_size()
            background = pygame.Surface(
                (width, height), pygame.SRCALPHA, pygame.RESIZABLE
            )
            screen.blit(background, (0, 0))
            pygame.display.flip()
            run_number = 0
            pygame.display.set_mode((0, 0), FULLSCREEN)
            screen.fill((0, 0, 0))
            if len(list_songs) > 0:
                filename = list_songs[jjj]
            wav = wavio.read(filename)
            rsleft = wav.data[:, 0]
            rsright = wav.data[:, 1]

            samp_rate = wav.rate

            lmax = max(rsleft)
            rmax = max(rsright)
            absmax = max(lmax, rmax)

            clock = pygame.time.Clock()

            pygame.mixer.music.load(filename)
            pygame.mixer.music.play(0)

            delta_freq = samp_rate / NFFT

            done = False
            pause = False
            while not done:

                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:

                        pygame.mixer.music.pause()

                        screen = pygame.display.set_mode(
                            (900, 500), pygame.NOFRAME
                        )
                        pygame.display.update()
                        layout2 = [
                            [
                                sg.Text(
                                    "Select folder or single WAV file"
                                )
                            ],
                            [
                                sg.Checkbox(
                                    "Restart Player", default=False
                                )
                            ],
                            [
                                sg.Checkbox("Reverse Direction", default=False),
                                sg.Checkbox("Higher Sensitivity", default=True),
                                sg.Checkbox("Greater Separation", default=True),
                            ],
                            [sg.Text("Chunk Size ", size=(10,1)), 
                             sg.Input(NFFT,size=(7,1)), 
                             sg.Text("High values boost res of low freqs",
                             size=(36,1)) ],
                            [
                            sg.Text("Color Scheme :", size=(10,1)),
                            sg.Rad('Blue Green Red', 1, default=True), 
                            sg.Rad('Red Yellow Blue', 1, default=False),
                            sg.Rad('Blue White Red', 1, default=False)
                            ], 
                            [sg.Button("OK"), sg.Button("QUIT")],
                            ]
                        window2 = sg.Window("Settings", layout2)
                        event, values = window2.read()
                        if event == "QUIT" or event == sg.WIN_CLOSED:
                            window2.close()
                            pygame.mixer.music.stop()
                            pygame.quit()
                            sys.exit()
                            main_loop = False
                            do_loop = False
                            break
                        window2.close()
                        restrt, reverse, hi_sens, hi_sep, NFFT, r1,r2,r3 = (
                            values[0],
                            values[1],
                            values[2],
                            values[3],
                            int(values[4]),
                            values[5],
                            values[6],
                            values[7]                           
                        )
                        if r1:
                            color_map=0
                        elif r2:
                            color_map=1
                        else:
                            color_map=2
                        pygame.display.quit()
                        pygame.init()
                        screen = pygame.display.set_mode(
                            (0, 0), pygame.FULLSCREEN
                        )
                        screen.blit(background, (0, 0))
                        pygame.display.flip()
                        run_number = 0
                        pygame.display.set_mode((0, 0), FULLSCREEN)
                        screen.fill((0, 0, 0))
                        pygame.mixer.music.unpause()

                gp = pygame.mixer.music.get_pos()

                posit = int(TUNE_T + samp_rate * gp / 1000)

                stopx = posit + NFFT
                if stopx > len(rsleft):
                    stopx = len(rsleft)
                if posit < 0:
                    posit = 0
                    stopx = NFFT

                chunk_left = rsleft[posit:stopx]
                chunk_right = rsright[posit:stopx]
                thismax = max(max(chunk_left), max(chunk_right))
                rel_vol = thismax / absmax
                if thismax>0:
                    try:
                        colors_255 = get_stereo_amps(
                            chunk_left,
                            chunk_right,
                            delta_freq,
                            rel_vol,
                            hi_sens,
                            hi_sep,
                            samp_rate,
                            color_map
                        )
                    except:
                        print("exception calling colors_255")
                        done = True
                    screen.fill((0, 0, 0))
    
                    for note in range(1, 89):
                        try:
                            draw_arc(note, colors_255[note - 1], reverse)
                        except:
                            print("exception running main program")
                            pygame.mixer.music.stop()
                            done = True
                    screen.blit(background, (0, 0))
                    pygame.display.flip()
                    if run_number == 0:
                        pygame.display.set_mode((0, 0), FULLSCREEN)
                        width, height = screen.get_size()
                        x_0 = width / 2
                        y_0 = height / 2
                        run_number = 1
                    clock.tick(40)
                if not pygame.mixer.music.get_busy():
                    done = True
                if restrt:
                    menu_num = 0
                    done = True
                    do_loop = False
                    file_path = ""
                    folder_path = ""
                    list_songs = []
                    pygame.quit()
                    break
            if jjj==Num_wav-1 and not restrt:
                do_loop=False
                main_loop=False
                break

pygame.quit()
sys.exit()
