import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


subtitle = ['steps of completing tasks', 'number of wall crashes', 'number of conflicts']

N = 100

mffp = np.load('./mffp_exp3.npy')
jsfp = np.load('./jsfp_exp3.npy')

data = [mffp[0], jsfp[0], mffp[1], jsfp[1], mffp[2], jsfp[2]]

fig, ax1 = plt.subplots(3, 1, figsize=(16, 9))
fig.canvas.set_window_title('A Boxplot of the Inventory Game')
fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
fig.suptitle('The Boxplot of Experiments Statistics', fontsize=20, fontweight='bold')

bp1 = ax1[0].boxplot(data[:2], notch=0, sym='+', vert=0, whis=1.5, positions=[0.07, 0.33])
ax1[0].set_title(subtitle[0], loc='center', fontdict={'fontsize': 8, 'fontweight': 'bold'})
plt.setp(bp1['boxes'], color='black')
plt.setp(bp1['whiskers'], color='black')
plt.setp(bp1['fliers'], color='red', marker='+')
boxColors = ['palegreen', 'royalblue']
numBoxes = 2
medians = list(range(numBoxes))
for i in range(numBoxes):
    box = bp1['boxes'][i]
    boxX = []
    boxY = []
    for j in range(5):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
    boxCoords = np.column_stack([boxX, boxY])
    # Alternate between Dark Khaki and Royal Blue
    k = i % 2
    boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
    ax1[0].add_patch(boxPolygon)
    # Now draw the median lines back over what we just filled in
    med = bp1['medians'][i]
    medianX = []
    medianY = []
    for j in range(2):
        medianX.append(med.get_xdata()[j])
        medianY.append(med.get_ydata()[j])
        ax1[0].plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
    # Finally, overplot the sample averages, with horizontal alignment
    # in the center of each box
    ax1[0].plot([np.average(data[:2][i])], [np.average(med.get_ydata())],
             color='w', marker='*', markeredgecolor='k')
ax1[0].set_xlim([28, 100])
ax1[0].set_ylim([-0.1, 0.5])
plt.setp(ax1[0], yticks=[])

bp2 = ax1[1].boxplot(data[2:4], notch=0, sym='+', vert=0, whis=1.5, positions=[0.07, 0.33])
ax1[1].set_title(subtitle[1], loc='center', fontdict={'fontsize': 8, 'fontweight': 'bold'})
plt.setp(bp2['boxes'], color='black')
plt.setp(bp2['whiskers'], color='black')
plt.setp(bp2['fliers'], color='red', marker='+')
boxColors = ['palegreen', 'royalblue']
numBoxes = 2
medians = list(range(numBoxes))
for i in range(numBoxes):
    box = bp2['boxes'][i]
    boxX = []
    boxY = []
    for j in range(5):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
    boxCoords = np.column_stack([boxX, boxY])
    # Alternate between Dark Khaki and Royal Blue
    k = i % 2
    boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
    ax1[1].add_patch(boxPolygon)
    # Now draw the median lines back over what we just filled in
    med = bp2['medians'][i]
    medianX = []
    medianY = []
    for j in range(2):
        medianX.append(med.get_xdata()[j])
        medianY.append(med.get_ydata()[j])
        ax1[1].plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
    # Finally, overplot the sample averages, with horizontal alignment
    # in the center of each box
    ax1[1].plot([np.average(data[2:4][i])], [np.average(med.get_ydata())],
             color='w', marker='*', markeredgecolor='k')
ax1[1].set_xlim([-2, 20])
ax1[1].set_ylim([-0.1, 0.5])
plt.setp(ax1[1], yticks=[])

bp3 = ax1[2].boxplot(data[4:], notch=0, sym='+', vert=0, whis=1.5, positions=[0.07, 0.33])
ax1[2].set_title(subtitle[2], loc='center', fontdict={'fontsize': 8, 'fontweight': 'bold'})
plt.setp(bp3['boxes'], color='black')
plt.setp(bp3['whiskers'], color='black')
plt.setp(bp3['fliers'], color='red', marker='+')
boxColors = ['palegreen', 'royalblue']
numBoxes = 2
medians = list(range(numBoxes))
for i in range(numBoxes):
    box = bp3['boxes'][i]
    boxX = []
    boxY = []
    for j in range(5):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
    boxCoords = np.column_stack([boxX, boxY])
    # Alternate between Dark Khaki and Royal Blue
    k = i % 2
    boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
    ax1[2].add_patch(boxPolygon)
    # Now draw the median lines back over what we just filled in
    med = bp2['medians'][i]
    medianX = []
    medianY = []
    for j in range(2):
        medianX.append(med.get_xdata()[j])
        medianY.append(med.get_ydata()[j])
        ax1[2].plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
    # Finally, overplot the sample averages, with horizontal alignment
    # in the center of each box
    ax1[2].plot([np.average(data[4:][i])], [np.average(med.get_ydata())],
             color='w', marker='*', markeredgecolor='k')
ax1[2].set_xlim([-5, 60])
ax1[2].set_ylim([-0.1, 0.5])
plt.setp(ax1[2], yticks=[])

# Finally, add a basic legend
fig.text(0.9, 0.02, 'Results of MFFP',
         backgroundcolor=boxColors[0], color='black', weight='roman',
         size='x-small')
fig.text(0.9, 0.04, 'Results of JSFP',
         backgroundcolor=boxColors[1],
         color='white', weight='roman', size='x-small')
fig.text(0.9, 0.06, '*', color='white', backgroundcolor='k',
         weight='roman', size='x-small')
fig.text(0.905, 0.06, ' Average Value', color='black', weight='roman',
         size='x-small')

plt.show()
