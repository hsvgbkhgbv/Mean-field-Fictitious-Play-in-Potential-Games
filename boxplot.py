import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy.stats import *

subtitle = ['number of steps to complete the task', 'number of iters to complete each situation']

mffp = np.load('./mffp_exp3_seed3.npy')
jsfp = np.load('./jsfp_exp3_seed3.npy')

mffp_ = np.load('./mffp_exp3_iter3.npy')
jsfp_ = np.load('./jsfp_exp3_iter3.npy')

data = [mffp[0], jsfp[0], mffp_, jsfp_]

fig, ax1 = plt.subplots(2, 1, figsize=(30, 5))
fig.canvas.set_window_title('A Boxplot of the Inventory Game')
fig.subplots_adjust(left=0.00, right=1., top=0.92, bottom=0.15, hspace=0.4)
# fig.suptitle('The Boxplot of Experiments Statistics', fontsize=14, fontweight='bold')

bp1 = ax1[0].boxplot(data[:2], notch=0, sym='+', vert=0, whis=1.5, positions=[0.12, 0.28], widths=.1)
ax1[0].set_title(subtitle[0], loc='center', fontdict={'fontsize': 20, 'fontweight': 'bold'})
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
ax1[0].set_xlim([35, 80])
ax1[0].set_ylim([0, 0.4])
ax1[0].tick_params(labelsize=16)
plt.setp(ax1[0], yticks=[])

bp2 = ax1[1].boxplot(data[2:4], notch=0, sym='+', vert=0, whis=1.5, positions=[0.12, 0.28], widths=.1)
ax1[1].set_title(subtitle[1], loc='center', fontdict={'fontsize': 20, 'fontweight': 'bold'})
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
ax1[1].set_xlim([200, 750])
ax1[1].set_ylim([0, 0.4])
ax1[1].tick_params(labelsize=16)
plt.setp(ax1[1], yticks=[])

# bp3 = ax1[2].boxplot(data[4:], notch=0, sym='+', vert=0, whis=1.5, positions=[0.07, 0.33])
# ax1[2].set_title(subtitle[2], loc='center', fontdict={'fontsize': 6, 'fontweight': 'bold'})
# plt.setp(bp3['boxes'], color='black')
# plt.setp(bp3['whiskers'], color='black')
# plt.setp(bp3['fliers'], color='red', marker='+')
# boxColors = ['palegreen', 'royalblue']
# numBoxes = 2
# medians = list(range(numBoxes))
# for i in range(numBoxes):
#     box = bp3['boxes'][i]
#     boxX = []
#     boxY = []
#     for j in range(5):
#         boxX.append(box.get_xdata()[j])
#         boxY.append(box.get_ydata()[j])
#     boxCoords = np.column_stack([boxX, boxY])
#     # Alternate between Dark Khaki and Royal Blue
#     k = i % 2
#     boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
#     ax1[2].add_patch(boxPolygon)
#     # Now draw the median lines back over what we just filled in
#     med = bp2['medians'][i]
#     medianX = []
#     medianY = []
#     for j in range(2):
#         medianX.append(med.get_xdata()[j])
#         medianY.append(med.get_ydata()[j])
#         ax1[2].plot(medianX, medianY, 'k')
#         medians[i] = medianY[0]
#     # Finally, overplot the sample averages, with horizontal alignment
#     # in the center of each box
#     ax1[2].plot([np.average(data[4:][i])], [np.average(med.get_ydata())],
#              color='w', marker='*', markeredgecolor='k')
# ax1[2].set_xlim([-5, 60])
# ax1[2].set_ylim([-0.1, 0.5])
# ax1[2].tick_params(labelsize=6)
# plt.setp(ax1[2], yticks=[])

# Finally, add a basic legend
fig.text(0.88, 0.03, 'Results of MFFP',
         backgroundcolor=boxColors[0], color='black', weight='roman',
         size=16)
fig.text(0.76, 0.03, 'Results of JSFP',
         backgroundcolor=boxColors[1],
         color='white', weight='roman', size=16)
fig.text(0.73, 0.03, '*', color='white', backgroundcolor='k',
         weight='roman', size=16)
fig.text(0.621, 0.03, ' Average Value', color='black', weight='roman',
         size=16)

plt.show()

twosample_results = ttest_ind(mffp[0], jsfp[0])
print ('This is the results of steps: {}.'.format(twosample_results))
twosample_results = ttest_ind(mffp[1], jsfp[1])
print ('This is the results of wall crashes: {}.'.format(twosample_results))
twosample_results = ttest_ind(mffp[2], jsfp[2])
print ('This is the results of conflicts: {}.'.format(twosample_results))
twosample_results = ttest_ind(mffp_, jsfp_)
print ('This is the results of iters: {}.'.format(twosample_results))
