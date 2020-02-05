###############################################################################
######################## WUHAN Coro patient prediction ########################

# The prediction is achieved by applying the Geometric series, along with a 
# recrusive algorithm. In this case, each person who get infected by this virus,
# will go through a 8 days period (T_period):(referred by thesis in 'Lancet')
#                                 3 days generate virus inside their body (T_delay)
#                                 5 days spread to other people (T_infect)
# Then after this 8-days period, this patient will go to hospital and no longer
# be a infection source.

###############################################################################

# Assumption: [1]. You might meet 60 diffreent people in a day.
#             [2]. You have 8.5% talk to them closely.
#             [3]. During those closely talk, you have 12% infect them.
#             [4]. Values assumed above is in line 44-46, and of course you can 
#                  tune them to see how these values influence the final prediction.
# Anyway, you will see how sensitive this final prediction when you tune these hyparameters

# In[1].Infection prediction 
import numpy as np
import date_transfer as dtr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Data preparation
Base = '2019-12-01'
wuhan_block_time = '2020-01-23'

# Get information from the thesis in 'Lancet'
T_incu = 8                     # Incubation period, [Day]
T_period = T_incu              # How many days that one person go into the hospital after getting infection, [Day]
T_delay = 3                    # How many days can be infectious after getting the virus, [Day]
T_infect = T_period - T_delay  # How many days that one patient can be infectious, [Day]


# Patient zero
nr_patient_zero = 1      # [01.12.2019]

###############################################################################
# Hyperparameters that can be tuned for this model.
# The hyperparameters below are very sensitive in the final prediction.
nr_people_meet_a_day = 60
percent_close_talking = 0.085
percent_infect = 0.12

# infection coe [q]
# This ratio is only valid until [wuhan_block_time]
q = nr_people_meet_a_day*percent_close_talking*percent_infect
###############################################################################
# Weight & ratio initialization
def weight_initialize(n2, q, nr_patient_zero):
    weight = {}
    ratio = {}
    length_1 = T_infect + (T_period - T_infect + 1)*(1 - 1)
    weight['layer' + str(1)] = np.ones(length_1)*nr_patient_zero
    ratio['layer' + str(1)] = np.ones(length_1)*q
    for i in range(2, n2 + 1):
        length = T_infect + (T_period - T_infect + 1)*(i - 1)
        weight['layer' + str(i)] = np.zeros(length)
        ratio['layer' + str(i)] = np.ones(length)*np.power(q, i)
    
    return weight, ratio


#####################################################
## Here is the fucking critical part of this model ## 
#####################################################
# Weights calculation basing on the model assumption
# The mathematical theory behind this [unique] structure, can be found in 'READ_ME', 
# or you can ask me by yourself directly. Actually I prefer the later one, since do
# the explaination in right here will be another torment for me.
def weight_calculate(weight, n2):
    for j in range(2, n2 + 1):
        for k in range(len(weight['layer' + str(j)])):
            if k <= T_infect-1:
                for ii in range(k+1):
                    weight['layer' + str(j)][k] += weight['layer' + str(j-1)][ii]
            else:
                if k >= len(weight['layer' + str(j-1)]):
                    weight['layer' + str(j)][k] = weight['layer' + str(j)][k-1] - \
                    weight['layer' + str(j-1)][k-T_infect]
                else:
                    weight['layer' + str(j)][k] = weight['layer' + str(j)][k-1] - \
                    weight['layer' + str(j-1)][k-T_infect] + weight['layer' + str(j-1)][k]
                    
    return weight



infection = []  # The infection accumulation going through time.
day_from_base = int(dtr.date_transfer(wuhan_block_time, Base))   # How many days away from 2019-12-01 
observing_starting_point = 7                                     # before blocking the city.
time_series = range(observing_starting_point, day_from_base)

for iii in time_series:

    n1 = (iii - 1)//(T_period - 1)    # How many layers have been finished
    n2 = (iii - 1)//T_delay           # How many layers have been started

    weight, ratio = weight_initialize(n2, q, nr_patient_zero)       
    weight_calculated = weight_calculate(weight, n2)
    
    
    # Calculate the geometric series                  
    multiply = {}
    for jj in range(1, n2 + 1):
        multiply['layer' + str(jj)] = weight['layer' + str(jj)]*ratio['layer' + str(jj)]
        
            
    # Now check the schedule of un-finished layers
    # Below the starting day represents when this layer get started for infection.
    starting_day = np.zeros(n2)
    length_of_each_layer = np.zeros(n2)
    for i in range(n2):
        starting_day[i] = T_delay*(i+1) + 1
        length_of_each_layer[i] = T_infect + (T_period - T_infect + 1)*(i-1+1)  # How many days this layer stands for
        
        
    length_take_from_multiply_4_unfinished = np.zeros(n2, dtype = int)
    for j in range(n1, n2):
        length_take_from_multiply_4_unfinished[j] = int(iii - starting_day[j])
        
    
    # Now calculate the how many people get infected until 'wuhan_block_time = '2020-01-23''
    # Initialize the infected people number by the 'number of patient zero'
    people_get_infected = nr_patient_zero
    
    # First of all, calculate the layers who are already finished
    for k in range(n1):
        people_get_infected += np.sum(multiply['layer' + str(k+1)])
    
    # Then, plus those layers who have not finished
    for kk in range(n1+1, n2+1):
        people_get_infected += np.sum(multiply['layer' + str(kk)][:(kk-1)])
    
    infection.append(people_get_infected)

R0 = q*T_infect
print('')
print('Calculating from 2019-12-01 to the time below,')
print('when Wuhan has been blocked:')
print('###########')
print(wuhan_block_time)
print('###########')
print('')
print('PREDICTION:')
print('R0 value before taking control in 2020-01-23: %3f' % R0, str('per person'))
print('There might be #%d# people got infected until 2020-01-23' % int(infection[-1]))


# In[2].Infection dynamic plot
# PLot as a gif 
plt.ion()
plt.figure(figsize=(12,8))
plt.title('Infection through time, when R0 = %3f, and %d patient_zero' % (R0, nr_patient_zero), fontsize = 20)
plt.xlabel('Days away from 2019-12-01', fontsize = 23)
plt.ylabel('Infection population', fontsize = 23)
t_list = []
result_list = []
t = 0
while t < len(time_series):
    t_list = time_series[t]
    result_list = infection[t]
    plt.plot(t_list, result_list,c='r',ls='-', marker='o', mec='r',mfc='w')
    #plt.plot(t, np.sin(t), 'o')
    t+=1
    plt.pause(0.1)
    
plt.plot(time_series, infection)
plt.vlines(time_series[-1], 0, (infection[-1] + 6000), colors = "k", linestyles = "dashed")
