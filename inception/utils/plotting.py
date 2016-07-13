import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from os import listdir

# PLOTTING
def plot_all_runs(results, window=100, two_panels=False,
                  sync_seconds_per_batch=None, async_seconds_per_batch=None,
                  marker_dict=None, mark_every=10, axis=[None,None,None,None],
                CG=None):
    MIN_MOM = -0.3
    MAX_MOM = 0.9
    STEP_MOM = 0.3

    cmap=plt.get_cmap('coolwarm')
    
    def color_from_mom(mom):
        return cmap((float(mom)-MIN_MOM)/(MAX_MOM-MIN_MOM))

    def lwfn(LR):
        return np.log(1.1+1e3*float(LR))

    # Using contourf to provide my colorbar info, then clearing the figure
    Z = [[0,0],[0,0]]
    #levels = np.linspace(MIN_MOM,MAX_MOM,100)
    levels = np.arange(MIN_MOM-STEP_MOM/2.0,MAX_MOM+1.5*STEP_MOM,STEP_MOM)
    ticks = np.arange(MIN_MOM,MAX_MOM+STEP_MOM,STEP_MOM)
    CS3 = plt.contourf(Z, levels, cmap=cmap)
    plt.clf()

    if two_panels:
        f,(ax1,ax2) = plt.subplots(1,2,figsize=(10,8), sharey=True, sharex=True)
    else:
        f,ax1 = plt.subplots(1,1,figsize=(8,6))

    if marker_dict is None:
        marker_dict = {'0.0005':'s', '0.001':'o', '0.005':'v'}

    if (not sync_seconds_per_batch is None) and (not async_seconds_per_batch is None):
        assert CG is not None,'Need number of CGs to plot over time'
        plot_over_time = True
    else:
        plot_over_time = False


    for folder, loss in results.items():
        LR, mom = params_from_folder_name(folder)
        if len(loss)<window:
            print('WARNING: Run shorter than window ('+str(window)+'): '+folder)
            continue
     
        if "True" in folder:
            avg = moving_average2(loss,window)
            if plot_over_time:
                ax1.plot(sync_seconds_per_batch * np.arange(len(avg)) /(60.0*CG), avg,
                         linewidth=lwfn(LR), c=color_from_mom(mom), markersize=10, marker = marker_dict[LR],
                         markevery=mark_every, label='LR='+str(LR)+', u='+str(mom))
                ax1.set_xlabel('Minutes into run')
            else:
                ax1.plot(np.arange(len(avg)), avg,
                         linewidth=lwfn(LR), c=color_from_mom(mom), markersize=10, marker = marker_dict[LR],
                         markevery=mark_every, label='LR='+str(LR)+', u='+str(mom))
                ax1.set_xlabel('Number of steps since snapshot')

    #ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if len(results)>15:
        ax1.legend(bbox_to_anchor=(0.0, -0.09), loc=2, borderaxespad=0.)
    else:
        ax1.legend(loc='best')
    ax1.grid()

    #ax1.axis([None,None,5.14,5.25]);
    ax1.set_ylabel('Smoothed training loss');
    if two_panels:
        ax1.set_title('Synchronous');
            
    #plt.figure(figsize=(5,8))

    if not two_panels:
        ax2=ax1
            
    for folder, loss in results.items():
        LR, mom = params_from_folder_name(folder)
        if len(loss)<window:
            print('WARNING: Run shorter than window ('+str(window)+'): '+folder)
            continue

        if "False" in folder:
            avg = moving_average2(loss,window)

            if plot_over_time:
                ax2.plot(async_seconds_per_batch * np.arange(len(avg)) /(60.0*CG), avg, '--',
                         linewidth=lwfn(LR), c=color_from_mom(mom), markersize=10, marker = marker_dict[LR],
                         markevery=mark_every, label='LR='+str(LR)+', u='+str(mom))
                ax2.set_xlabel('Minutes into run')
            else:
                ax2.plot(np.arange(len(avg)), avg, '--',
                         linewidth=lwfn(LR), c=color_from_mom(mom), markersize=10, marker = marker_dict[LR],
                         markevery=mark_every, label='LR='+str(LR)+', u='+str(mom))
                ax2.set_xlabel('Number of steps since snapshot')
            
    if len(results)>15:
        ax2.legend(bbox_to_anchor=(0.0, -0.09), loc=2, borderaxespad=0.)
    else:
        ax2.legend(loc='best')

    if two_panels:
        ax2.grid()
        ax2.set_title('Asynchronous');

    ax1.axis(axis)


    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.03, 0.7])
    cbar = f.colorbar(CS3, cax=cbar_ax, ticks=ticks)
    cbar.ax.set_ylabel('Momentum')
    #cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')

    return f

def plot_loss_contour(loss_results, window, sync=True):

    x=[]
    y=[]
    z=[]
       

    for folder, loss in loss_results.items():
        if (sync and "True" in folder) or (not sync and "False" in folder):
            if len(loss)<2*window:
                print('WARNING: Run shorter than window ('+str(window)+'): '+folder)
                continue
            if any(np.isnan(loss)):
                print('WARNING: NaN value found in: '+folder)
                continue
            LR, mom = params_from_folder_name(folder)
            avg = moving_average2(loss,window)
            assert not np.isnan(avg[-1])

            x.append(float(mom))
            y.append(float(LR))
            z.append(avg[-1])


    min_z = np.min(z)

    plt.figure()
    plt.tricontourf(x,np.log10(y),z,
                   levels=np.linspace(min_z, min_z*1.1, 20))
    #plt.tricontourf(x,np.log10(y),z,
    #               levels=np.linspace(0.32, 0.42, 20))
                #, norm=plt.Normalize(vmax=minz*1.4, vmin=minz))
    plt.plot(x,np.log10(y),'ko')
    plt.ylabel('log10(LR)')
    plt.xlabel('Momentum')
    plt.colorbar()
    if sync:
        plt.title('Synchronous');
    else:
        plt.title('Asynchronous');

def plot_se_calculation(loss_results, window):
    best_sync, best_async, best_sync_name, best_async_name = get_best_for_each_mode(loss_results, window)

    s = best_sync.copy()
    a = best_async.copy()

    min_s = min(s[window:])
    min_a = min(a[window:])
    target = max(min_s, min_a)

    fig = plt.figure()

    plt.plot(s, linewidth=3)
    plt.plot(a, '--', c='orange', linewidth=3)
    plt.plot([0,np.max([len(s),len(a)])],2*[target],'--k')
    plt.xlabel('Number of steps since snapshot')
    plt.ylabel('Smoothed loss')
    plt.legend(['Sync: ' + best_sync_name, 'Async: ' + best_async_name, 'Evaluation level'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.title('Best run for each mode, loss used for SE calculation')
    plt.grid()

    return fig

def plot_momentum_dependence(loss_results, window):
    dict_of_winners = get_best_for_each_momentum(loss_results,window)

    sync_x = []
    sync_y = []
    async_x = []
    async_y = []

    sync_list = []
    async_list = []

    for folder, loss in dict_of_winners.items():
        avg = moving_average2(loss,window)
        LR, mom = params_from_folder_name(folder)
        if "True" in folder:
            sync_list.append((float(mom),avg[-1]))
        elif "False" in folder:
            async_list.append((float(mom),avg[-1]))


    sync_list.sort()

    sync_x = [el[0] for el in sync_list]
    sync_y = [el[1] for el in sync_list]

    async_list.sort()

    async_x = [el[0] for el in async_list]
    async_y = [el[1] for el in async_list]

    f=plt.figure()

    LW=3
    plt.plot(sync_x, sync_y, linewidth=LW, label='Synchronous', c='blue')
    plt.plot(async_x, async_y, '--', linewidth=LW, label='Asynchronous', c='orange')
    plt.xlabel('Momentum')
    plt.ylabel('Smoothed training loss at end (- edge effects)')
    plt.grid()
    plt.legend(loc='best')

    return f

 
def plot_winners_snr(loss_results, window):
    best_sync, best_async, best_sync_name, best_async_name = get_best_for_each_mode(loss_results, window)

    s = best_sync.copy()
    a = best_async.copy()

    fig = plt.figure()

    plt.plot(loss_results[best_sync_name],'r',alpha=0.2)
    plt.plot(s,'r',linewidth=4)
    plt.plot(loss_results[best_async_name],'g',alpha=0.2)
    plt.plot(a,'--g',linewidth=4)

    plt.xlabel('Number of steps after snapshot')
    plt.ylabel('Smoothed loss')
    plt.legend(['Best Sync: measurement', 'Best Sync: smoothed','Best Async: measurement', 'Best Async: smoothed'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.title('Measured losses, smoothed signals used in SE plotting')
    plt.grid()

    return fig

def plot_se_he(loss_results, window, M, sync_seconds_per_batch,
                 async_seconds_per_batch):
    best_sync, best_async, best_sync_name, best_async_name = get_best_for_each_mode(loss_results, window)

    s = best_sync.copy()
    a = best_async.copy()

    min_s = min(s[window:])
    min_a = min(a[window:])
    target = max(min_s, min_a)

    async_meets_target = np.argmax(a <= target)
    sync_meets_target = np.argmax(s <= target)

    fig = plt.figure()

    SE = np.array([sync_meets_target,async_meets_target])/float(sync_meets_target)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4))
    ax1.plot([1,M],SE, '-s')
    ax1.set_ylabel('Statistical Efficiency')
    ax1.set_xlabel('# compute groups')
    ax1.grid()
    ax1.axis([0.9,M+.1, 0, 2])

# Hardware efficiency
    sync_time_per_iter = sync_seconds_per_batch

    async_time_per_iter = async_seconds_per_batch

    HE = np.array([sync_time_per_iter,async_time_per_iter])/float(sync_time_per_iter)

    ax2.plot([1,M],HE, '-s')
    ax2.set_ylabel('Hardware Efficiency')
    ax2.set_xlabel('# compute groups')
    ax2.grid()
    ax2.axis([0.9,M+.1, 0, 2])


    ax3.plot([1,M],np.multiply(HE,SE), '-s')
    ax3.set_ylabel('Relative Time')
    ax3.set_xlabel('# compute groups')
    ax3.grid()
    ax3.axis([0.9,M+.1, 0, 2]);


    return fig
    
    

# SELECTION
def get_best_for_each_mode(results, window):
    best_async = None
    best_sync = None
    async_keeper = None
    sync_keeper = None

    for folder, loss in results.items():
        if len(loss)<window:
            print('WARNING: Run shorter than window ('+str(window)+'): '+folder)
            continue
     
        if "True" in folder:
            avg = moving_average2(loss,window)
            if avg[-1] < sync_keeper or sync_keeper is None:
                sync_keeper = avg[-1]
                best_sync = avg
                best_sync_name = folder

    for folder, loss in results.items():
        if len(loss)<window:
            print('WARNING: Run shorter than window ('+str(window)+'): '+folder)
            continue

        if "False" in folder:
            avg = moving_average2(loss,window)

            if avg[-1] < async_keeper or async_keeper is None:
                async_keeper = avg[-1]
                best_async = avg
                best_async_name = folder
    return best_sync, best_async, best_sync_name, best_async_name

def get_best_for_each_momentum(results, window):
    best_dict_sync={}
    best_dict_async={}
     

    for folder, loss in results.items():
        if len(loss)<window:
            print('WARNING: Run shorter than window ('+str(window)+'): '+folder)
            continue
        LR, mom = params_from_folder_name(folder)
        avg = moving_average2(loss,window)


        if "True" in folder:
            if mom not in best_dict_sync:   
                best_dict_sync[mom] = (folder, avg[-1])
            elif avg[-1] < best_dict_sync[mom][1]:
                best_dict_sync[mom] = (folder, avg[-1])
        elif "False" in folder:
            if mom not in best_dict_async:   
                best_dict_async[mom] = (folder, avg[-1])
            elif avg[-1] < best_dict_async[mom][1]:
                best_dict_async[mom] = (folder, avg[-1])
    
    dict_of_winners={}
    list_of_winner_names=[]

    for (key,value) in best_dict_sync.iteritems():
        list_of_winner_names.append(value[0])
        dict_of_winners[value[0]] = results[value[0]]
    for (key,value) in best_dict_async.iteritems():
        list_of_winner_names.append(value[0])
        dict_of_winners[value[0]] = results[value[0]]

    return dict_of_winners

# SMOOTHING
def moving_average2(signal, window_size=100):
    window = np.ones(int(window_size))/float(window_size)
    conv = np.convolve(signal, window, 'same')
    conv = np.convolve(conv, window, 'same')[:-window_size]
    conv[:window_size] = np.nan
    return conv

def moving_average1(signal, window_size=100):
    window = np.ones(int(window_size))/float(window_size)
    conv = np.convolve(signal, window, 'same')[:-window_size//2]
    conv[:window_size//2] = np.nan
    return conv

# LOADING RUN LOGS
def load_list_of_runs(list_of_runs, data_dir='.'):
    acc_results = {}
    loss_results = {}
    print 'Loading run logs:'
    print '----------------'
    # List files in data:
    for folder in list_of_runs:
        print folder
        acc_results[folder] = []
        loss_results[folder] = []
        for filename in listdir(data_dir+'/'+folder):
            if filename != 'ps.out':
                with open(data_dir+"/"+folder+"/"+filename, "r") as f:
                    for line in f.readlines():
                        if line.startswith("INFO:tensorflow:Step:"):
                            parts = line.split(" ")
                            step = int(parts[1][:-1])
                            if step > 5000:
                                break
                            accuracy = float(parts[3][:-1])
                            loss = float(parts[5])
                            acc_results[folder].append((step, accuracy))
                            loss_results[folder].append((step, loss))

    for key, val in acc_results.items():
        val.sort(key=lambda x: x[0])
        acc_results[key] = [y for x,y in val]
    for key, val in loss_results.items():
        val.sort(key=lambda x: x[0])
        loss_results[key] = [y for x,y in val]

    return loss_results, acc_results

def params_from_folder_name(folder):
    folderParts = folder.split('-')
    if folderParts[0]=='ShortRun':
        if folderParts[1]=='seed':
            if folderParts[3]=='30min':
                field_zero = 4
            else:
                field_zero = 3
        else:
            field_zero = 1
    elif folderParts[0]=='LongRun':
        if folderParts[2]=='v2':
            field_zero = 3
        else:
            field_zero = 2
    elif folderParts[0]=='expr2':
        field_zero = 1
    else:       
        field_zero = 2

    # Read LR and mom starting from index field_zero
    LR = folderParts[field_zero]
    if len(folderParts[field_zero+1])==0:
        mom = '-' + folderParts[field_zero+2]
    else:
        mom = folderParts[field_zero+1]
        
    return LR, mom


def params_from_folder_name_old(folder):
    folderParts = folder.split('-')
    if folderParts[0]=='ShortRun':
        LR = folderParts[3]
        if len(folderParts[4])==0:
            mom = '-' + folderParts[5]
        else:
            mom = folderParts[4]
    elif folderParts[0]=='LongRun':
        if folderParts[2]=='v2':
            field_zero = 3
        else:
            field_zero = 2
        LR = folderParts[field_zero]
        if len(folderParts[field_zero+1])==0:
            mom = '-' + folderParts[field_zero+2]
        else:
            mom = folderParts[field_zero+1]
    else:        
        LR = folderParts[2]
        if len(folderParts[3])==0:
            mom = '-' + folderParts[4]
        else:
            mom = folderParts[3]
        
    return LR, mom





