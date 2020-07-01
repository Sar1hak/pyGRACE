from __future__ import print_function
import matplotlib.pyplot as plt
import os

##### Plotting maps #####
savepath=os.path.join(os.getcwd(),'Plots')

def loss_plot(history):
    """
    Arguments:
        history: History of the trained model, used to find the 
        trained and valuated loss.
    """
    #plotting loss graph
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig(os.path.join(savepath,"Loss_Graph.jpg"))
    plt.show()


def monthlyplot(initial_month_avg, output_month_avg):
    """
    Arguments:
        initial_month_avg: Monthly avergae value of the tiff image
        before undergoing process.
        output_month_avg: Monthly average of the tiff file after the
        processing.
    """
    month=[i for i in range(1,13)]
    year=[i for i in range(2000,2019)]

    fig, axes = plt.subplots(1, 2))    
    # Plot Intial monthly graph
    for i in initial_month_avg:
        axes[0][0].plot(month, i)
	axes[0][0].set_title('Initial Monthly Avg. Rainfall vs Year')
    plt.legend(year, loc='upper right')

    # Plot Output monthly graph
    for i in output_month_avg:
        fig = plt.figure()
        axes[0][1].plot(month, i)
	axes[0][1].set_title('Final Monthly Avg. Rainfall vs Year')
    plt.legend(year, loc='upper right')
    plt.savefig(os.path.join(savepath,"Monthly_Rainfall_Graph.jpg"))    
    plt.show()


def yearlyplot(initial_year_avg, output_year_avg):
    """
    Arguments:
        initial_year_avg: Yearly avergae value of the tiff image
        before undergoing process.
        output_year_avg: Yearly average of the tiff file after the
        processing.
    """
    month=[i for i in range(1,13)]
    year=[i for i in range(2000,2019)]
    
	fig, axes = plt.subplots(1, 2)) 
    # Plot Intial monthly graph
    for i in initial_year_avg:
        fig = plt.figure()
        axes[0][0].plot(month, i)
	axes[0][0].set_title('Initial Yearly Avg. Rainfall vs Year')
    plt.legend(year, loc='upper right')
    plt.show()

    # Plot Output monthly graph
    for i in output_year_avg:
        fig = plt.figure()
        axes[0][1].plot(month, i)
    axes[0][1].set_title('Final Yearly Avg. Rainfall vs Year')
    plt.legend(year, loc='upper right')
    plt.savefig(os.path.join(savepath,"Yearly_Rainfall_Graph.jpg"))
    plt.show()
