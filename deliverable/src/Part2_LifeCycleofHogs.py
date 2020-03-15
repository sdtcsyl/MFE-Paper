# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:03:51 2020

@author: e0348815
"""

import matplotlib.pyplot as plt
import numpy as np

rooms=['Stage 1','Stage 2', 'Stage 3']
colors=['lightblue', 'lightblue', 'lightblue']

input_files=['hoglifecycle.txt']
day_labels=['Life Cycle of Hogs']


for input_file, day_label in zip(input_files, day_labels):
    fig=plt.figure(figsize=(10,5.89))
    for line in open(input_file, 'r'):
        data=line.split()
        print(data)
        event=data[-1]
        data=list(map(float, data[:-1]))
        room=data[0]-0.48
        start=data[1]+data[2]
        end=start+data[3]
        # plot event
        plt.fill_between([room, room+0.96], [start, start], [end,end], color=colors[int(data[0]-1)], edgecolor='k', linewidth=0.5)
        # plot beginning time
        plt.text(room+0.02, start ,'{0}'.format(int(start)), va='top', fontsize=12)
        plt.text(room+0.02, end ,'{0}'.format(int(end)), va='top', fontsize=12)
        # plot event name
        plt.text(room+0.48, (start+end)*0.5, event, ha='center', va='center', fontsize=14, fontweight='bold')

    # Set Axis
    ax=fig.add_subplot(111)
    #ax.yaxis.grid()
    ax.axvline(x=1.5, ymin=0, ymax=300,color='grey')
    ax.axvline(x=2.5, ymin=0, ymax=300,color='grey')
    #ax.xaxis.grid()
    ax.set_xlim(0.5,len(rooms)+0.5)
    ax.set_ylim(0, 290,10)
    ax.set_xticks(range(1,len(rooms)+1))
    ax.set_xticklabels(rooms)
    ax.set_ylabel('Hogs Weight (Pounds)')

    # Set Second Axis
    ax2=ax.twiny().twinx()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylim(ax.get_ylim())
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels(rooms)
    ax2.set_ylabel('Hogs Weight (Pounds)')

    plt.title(day_label,y=1.07)



