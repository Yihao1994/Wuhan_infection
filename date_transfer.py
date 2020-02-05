from datetime  import datetime
import time
import numpy as np

def date_transfer(date, Base):
    
    sec_for_a_day = 24*60*60   # How many secs there ae for a day

    # Transfer the base to sec
    Base = datetime.strptime(Base, '%Y-%m-%d')
    un_time_Base = time.mktime(Base.timetuple())
    
    # Transfer the date to sec
    date = datetime.strptime(date, '%Y-%m-%d')
    un_time_date = time.mktime(date.timetuple())
    day_from_base = (un_time_date - un_time_Base)/sec_for_a_day
    
    return day_from_base

